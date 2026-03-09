import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.google.gemini_live import GeminiLiveLLMService
from pipecat.services.google.gemini_live.llm import GeminiVADParams, InputParams
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

SYSTEM_PROMPT = """Sei Marco di GreenPower Italia, operatore outbound di un call center energia. Chiami in modo massivo da liste contatti. Sei veloce, diretto, e usi tutte le tecniche di vendita telefonica. Parla in italiano.

IMPORTANTE: Appena la conversazione inizia, presentati IMMEDIATAMENTE senza aspettare che l'utente parli. Dì subito: 'Buongiorno, sono Marco di GreenPower Italia, la chiamo per una proposta vantaggiosa sulla sua fornitura energetica. Ha un minuto?'"""


class LatencyLogger(FrameProcessor):
    """Logs timestamps for key events to measure latency."""

    def __init__(self):
        super().__init__()
        self._user_stop_talking = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._user_stop_talking = time.time()
            logger.info(f"[LATENCY] User said: '{frame.text}' at {self._user_stop_talking:.3f}")

        if isinstance(frame, TTSStartedFrame):
            now = time.time()
            if self._user_stop_talking:
                delta = now - self._user_stop_talking
                logger.info(f"[LATENCY] Bot starts speaking {delta:.3f}s after user stopped")
            else:
                logger.info(f"[LATENCY] Bot starts speaking (first utterance) at {now:.3f}")

        if isinstance(frame, TTSStoppedFrame):
            logger.info(f"[LATENCY] Bot stopped speaking at {time.time():.3f}")

        await self.push_frame(frame, direction)


async def on_user_idle(processor):
    logger.info("User silent for 8s, prompting")
    await processor.push_frame(
        LLMMessagesAppendFrame(
            messages=[
                {
                    "role": "user",
                    "content": "Il cliente è in silenzio, sollecitalo brevemente.",
                }
            ]
        )
    )


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-live-2.5-flash-native-audio.",
        voice_id="Puck",
        system_instruction=SYSTEM_PROMPT,
        params=InputParams(
            language=Language.IT,
            vad=GeminiVADParams(
                start_sensitivity="START_SENSITIVITY_HIGH",
                end_sensitivity="END_SENSITIVITY_HIGH",
                silence_duration_ms=100,
                prefix_padding_ms=50,
            ),
        ),
    )

    idle = UserIdleProcessor(timeout=8.0, callback=on_user_idle)
    latency = LatencyLogger()

    pipeline = Pipeline(
        [
            transport.input(),
            idle,
            latency,
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        t = time.time()
        logger.info(f"[LATENCY] Call connected at {t:.3f}")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Presentati subito, brevemente. Sei Marco di GreenPower Italia.",
                        }
                    ]
                )
            ]
        )
        logger.info(f"[LATENCY] Initial prompt queued at {time.time():.3f} (delta {time.time() - t:.3f}s)")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    _, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Incoming call: {call_data}")

    serializer = TelnyxFrameSerializer(
        stream_id=call_data["stream_id"],
        outbound_encoding=call_data["outbound_encoding"],
        inbound_encoding="PCMU",
        call_control_id=call_data["call_control_id"],
        api_key=os.getenv("TELNYX_API_KEY"),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
