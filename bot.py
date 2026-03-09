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
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from google.genai.types import ProactivityConfig
from pipecat.services.google.gemini_live.llm import (
    ContextWindowCompressionParams,
    GeminiVADParams,
    InputParams,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

SYSTEM_PROMPT = """YOU MUST RESPOND IN ITALIAN. YOU MUST RESPOND UNMISTAKABLY IN ITALIAN.

Sei Marco di GreenPower Italia, operatore outbound di un call center energia. Chiami in modo massivo da liste contatti. Sei veloce, diretto, e usi tutte le tecniche di vendita telefonica.

REGOLE:
- Risposte BREVI e DIRETTE, massimo 2 frasi per turno
- Non fare pause lunghe, rispondi subito
- Appena la conversazione inizia, presentati IMMEDIATAMENTE senza aspettare che l'utente parli
- Dì subito: 'Buongiorno, sono Marco di GreenPower Italia, la chiamo per una proposta vantaggiosa sulla sua fornitura energetica. Ha un minuto?'"""


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


_greeted = False


async def on_user_idle(processor):
    global _greeted
    if not _greeted:
        logger.info("First idle trigger — forcing greeting")
        _greeted = True
        msg = "Presentati subito, brevemente. Sei Marco di GreenPower Italia."
    else:
        logger.info("User silent, prompting")
        msg = "Il cliente è in silenzio, sollecitalo brevemente."
    await processor.push_frame(
        LLMMessagesAppendFrame(messages=[{"role": "user", "content": msg}])
    )


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm = GeminiLiveVertexLLMService(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        credentials=os.getenv("GOOGLE_VERTEX_CREDENTIALS"),
        model="gemini-live-2.5-flash-native-audio",
        voice_id="Puck",
        system_instruction=SYSTEM_PROMPT,
        inference_on_context_initialization=True,
        params=InputParams(
            temperature=0.7,
            language=Language.IT,
            vad=GeminiVADParams(
                start_sensitivity="START_SENSITIVITY_HIGH",
                end_sensitivity="END_SENSITIVITY_HIGH",
                silence_duration_ms=100,
                prefix_padding_ms=50,
            ),
            context_window_compression=ContextWindowCompressionParams(
                enabled=True,
            ),
            proactivity=ProactivityConfig(
                proactive_audio=True,
            ),
        ),
    )

    idle = UserIdleProcessor(timeout=1.5, callback=on_user_idle)
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
        logger.info(f"[LATENCY] Call connected at {time.time():.3f}")

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
