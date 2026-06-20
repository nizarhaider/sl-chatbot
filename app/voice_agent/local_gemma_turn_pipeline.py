import asyncio
import logging
import os
import time

import numpy as np
from av.audio.resampler import AudioResampler

logger = logging.getLogger(__name__)

GEMMA_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "").strip()
GEMMA_MODEL_REPO = os.environ.get(
    "GEMMA_MODEL_REPO",
    "google/gemma-4-12B-it-qat-q4_0-gguf",
).strip()
GEMMA_MODEL_FILENAME = os.environ.get("GEMMA_MODEL_FILENAME", "").strip()
GEMMA_MODEL_DIR = os.environ.get("GEMMA_MODEL_DIR", "").strip()
GEMMA_N_GPU_LAYERS = int(os.environ.get("GEMMA_N_GPU_LAYERS", "-1"))
GEMMA_CONTEXT_TOKENS = int(os.environ.get("GEMMA_CONTEXT_TOKENS", "4096"))
GEMMA_BATCH_TOKENS = int(os.environ.get("GEMMA_BATCH_TOKENS", "512"))
GEMMA_THREADS = int(os.environ.get("GEMMA_THREADS", "8"))
GEMMA_TEMPERATURE = float(os.environ.get("GEMMA_TEMPERATURE", "0.2"))
GEMMA_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMMA_MAX_OUTPUT_TOKENS", "160"))
GEMMA_PREWARM = os.environ.get("GEMMA_PREWARM", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

WHISPER_MODEL = "SPEAK-ASR/whisper-medium-si-merged"
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "sinhala").strip() or None
WHISPER_TASK = os.environ.get("WHISPER_TASK", "transcribe").strip()

TURN_INPUT_CHUNK_MS = max(20, int(os.environ.get("TURN_INPUT_CHUNK_MS", "40")))
TURN_INPUT_CHUNK_SIZE = (16000 * 2 * TURN_INPUT_CHUNK_MS) // 1000
TURN_SILENCE_THRESHOLD = int(os.environ.get("TURN_SILENCE_THRESHOLD", "1000"))
TURN_END_SILENCE_CHUNKS = max(2, int(os.environ.get("TURN_END_SILENCE_CHUNKS", "50")))
TURN_MIN_AUDIO_MS = max(200, int(os.environ.get("TURN_MIN_AUDIO_MS", "500")))

TURN_GREETING_DELAY_SECONDS = max(
    0.0,
    float(os.environ.get("TURN_GREETING_DELAY_SECONDS", "1.2")),
)
TURN_GREETING_PROTECTION_MAX_SECONDS = max(
    0.0,
    float(os.environ.get("TURN_GREETING_PROTECTION_MAX_SECONDS", "1.5")),
)

LOCAL_TURN_GREETING = os.environ.get(
    "LOCAL_TURN_GREETING",
    (
        "To speak in English, please say English. "
        "සිංහලෙන් කතා කිරීමට කරුණාකර සිංහල කියන්න. "
        "தமிழில் பேச தயவுசெய்து தமிழ் என்று சொல்லுங்கள்."
    ),
)

HOMELANDS_PROPERTIES = (
    "1. Horizon Residencies, Malabe: two-bedroom apartments from LKR 28 million, near schools and supermarkets. "
    "2. Lakeview Villas, Piliyandala: three-bedroom villas from LKR 48 million, garden, parking, and lake access. "
    "3. Green Acres, Kurunegala: ten-perch residential land from LKR 9.5 million, clear title, bank loans supported. "
    "4. Ocean Breeze Apartments, Dehiwala: one and two-bedroom units from LKR 32 million, sea view, ready soon."
)

HOMELANDS_LOCAL_SYSTEM_PROMPT = os.environ.get(
    "HOMELANDS_LOCAL_SYSTEM_PROMPT",
    (
        "You are a friendly assistant working for Homelands Properties. "
        "Help the customer with their property-related queries. "
        f"Use these mock properties only: {HOMELANDS_PROPERTIES} "
        "The caller has already heard a language-selection greeting asking them to say English, Sinhala, or Tamil. "
        "If the caller only picks a language, greet them briefly in that language and ask how you can help. "
        "Reply in the same language as the caller's latest speech unless they clearly ask to switch languages. "
        "If the caller message is unclear, garbled, partial, or you do not understand it, briefly say that you did not understand and ask them to repeat it. "
        "Do not repeat the language-selection greeting unless the caller is clearly starting over. "
        "Do not recommend a property, make up requirements, or schedule anything unless the caller actually asked about a property. "
        "When the customer asks about properties, recommend a suitable option and say that a Homelands consultant appointment has been scheduled for tomorrow at 10 AM. "
        "Keep each response brief, natural, and suitable for a phone call."
    ),
)

REALTIME_TTS_REF_AUDIO = "omnivoice-one-shot-dataset/chandeera-female-sample.ogg"
REALTIME_TTS_REF_TEXT = "ඔබතුමියගේ internet  connection එකේ ඇතිවී තිබෙන තාක්ෂණික දෝෂය පිළිබඳව මේවෙනකොටත් අපට වාර්තා වී තිබෙනවා. අපේ Technician කෙනෙක් ඉදිරි පැය විසිහතර ඇතුළත ඔබව visit කරලා ඔබේ ගැටලුට විසඳුමක් ලබාදේවි."

REALTIME_TTS_REF_LANGUAGE = os.environ.get("REALTIME_TTS_REF_LANGUAGE", "si")
REALTIME_TTS_NUM_STEPS = os.environ.get("REALTIME_TTS_NUM_STEPS", "12,12")
REALTIME_TTS_DEVICE = os.environ.get("REALTIME_TTS_DEVICE", "cuda:0")
REALTIME_TTS_DTYPE = os.environ.get("REALTIME_TTS_DTYPE", "float16")
REALTIME_TTS_DEBUG = os.environ.get("REALTIME_TTS_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REALTIME_TTS_PREWARM = os.environ.get("REALTIME_TTS_PREWARM", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class RealtimeOmniVoiceTTS:
    def __init__(self) -> None:
        self._stream = None
        self._sample_rate = 24000
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_stream)

    async def speak(self, text: str, on_audio_chunk) -> float:
        async with self._lock:
            chunks: list[bytes] = []

            def collect_and_forward(chunk: bytes) -> None:
                chunks.append(chunk)
                on_audio_chunk(chunk, self._sample_rate)

            started = time.perf_counter()
            await asyncio.to_thread(self._play, text, collect_and_forward)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            audio_bytes = sum(len(chunk) for chunk in chunks)
            audio_seconds = self._audio_duration_seconds(audio_bytes)
            logger.info(
                "RealtimeTTS complete: elapsed_ms=%.0f chars=%s audio_ms=%.0f chunks=%s",
                elapsed_ms,
                len(text),
                audio_seconds * 1000.0,
                len(chunks),
            )
            return audio_seconds

    def _play(self, text: str, on_audio_chunk) -> None:
        stream = self._get_stream()
        stream.feed(text)
        stream.play(muted=True, on_audio_chunk=on_audio_chunk)

    def _get_stream(self):
        if self._stream is not None:
            return self._stream

        from RealtimeTTS import OmniVoiceEngine, OmniVoiceVoice, TextToAudioStream
        import torch

        class PatchedOmniVoiceEngine(OmniVoiceEngine):
            def synthesize(self, text: str, sentence_count: int = 0) -> bool:
                super(OmniVoiceEngine, self).synthesize(text, sentence_count)

                if not self.current_voice:
                    return False

                current_num_steps = self._get_num_steps_for_sentence(sentence_count)

                try:
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        audio = self._model.generate(
                            language=self.current_voice.language,
                            text=text,
                            ref_audio=self.current_voice.ref_audio,
                            ref_text=self.current_voice.ref_text,
                            num_step=current_num_steps,
                            preprocess_prompt=self.preprocess_prompt,
                            postprocess_output=self.postprocess_output,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                    waveform = _normalize_omnivoice_waveform(audio)
                    audio_int16 = (
                        np.clip(waveform, -1.0, 1.0) * 32767
                    ).astype(np.int16).tobytes()
                    self.queue.put(audio_int16)
                    return True
                except Exception:
                    logger.exception("Patched OmniVoice synthesis failed for text: %s", text)
                    return False

        steps = [
            int(part.strip())
            for part in REALTIME_TTS_NUM_STEPS.split(",")
            if part.strip()
        ]

        voice = OmniVoiceVoice(
            name="homelands",
            ref_audio=REALTIME_TTS_REF_AUDIO,
            ref_text=REALTIME_TTS_REF_TEXT,
            language=REALTIME_TTS_REF_LANGUAGE,
        )

        engine = PatchedOmniVoiceEngine(
            voice=voice,
            device_map=REALTIME_TTS_DEVICE,
            dtype=getattr(torch, REALTIME_TTS_DTYPE),
            num_steps_schedule=steps or [12, 12],
            debug=REALTIME_TTS_DEBUG,
        )

        _, _, self._sample_rate = engine.get_stream_info()
        self._stream = TextToAudioStream(engine, muted=True)
        return self._stream

    def _audio_duration_seconds(self, audio_bytes: int) -> float:
        if audio_bytes <= 0 or self._sample_rate <= 0:
            return 0.0
        return (audio_bytes / 2) / self._sample_rate


def _normalize_omnivoice_waveform(audio) -> np.ndarray:
    waveform = audio
    if isinstance(waveform, (list, tuple)):
        waveform = waveform[0]
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    else:
        waveform = np.asarray(waveform)
    waveform = np.squeeze(waveform)
    if waveform.ndim > 1:
        waveform = waveform[0]
    return waveform.astype(np.float32, copy=False)


class LocalGemmaLLM:
    def __init__(self) -> None:
        self._llm = None
        self._lock = asyncio.Lock()

    async def prewarm(self) -> None:
        await asyncio.to_thread(self._get_llm)

    async def generate(self, transcript_text: str) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, transcript_text)

    def _generate_sync(self, transcript_text: str) -> str:
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": HOMELANDS_LOCAL_SYSTEM_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
            temperature=GEMMA_TEMPERATURE,
            max_tokens=GEMMA_MAX_OUTPUT_TOKENS,
        )
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        from llama_cpp import Llama

        model_path = self._resolve_model_path()
        logger.info(
            "Loading local Gemma model: path=%s n_gpu_layers=%s n_ctx=%s n_batch=%s",
            model_path,
            GEMMA_N_GPU_LAYERS,
            GEMMA_CONTEXT_TOKENS,
            GEMMA_BATCH_TOKENS,
        )
        started = time.perf_counter()
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=GEMMA_N_GPU_LAYERS,
            n_ctx=GEMMA_CONTEXT_TOKENS,
            n_batch=GEMMA_BATCH_TOKENS,
            n_threads=GEMMA_THREADS,
            verbose=False,
        )
        logger.info(
            "Local Gemma model loaded in %.0f ms",
            (time.perf_counter() - started) * 1000.0,
        )
        return self._llm

    def _resolve_model_path(self) -> str:
        if GEMMA_MODEL_PATH:
            return GEMMA_MODEL_PATH

        from huggingface_hub import HfApi, hf_hub_download

        filename = GEMMA_MODEL_FILENAME
        if not filename:
            files = HfApi().list_repo_files(GEMMA_MODEL_REPO)
            q4_files = [
                path
                for path in files
                if path.lower().endswith(".gguf") and "q4_0" in path.lower()
            ]
            if not q4_files:
                raise RuntimeError(
                    f"No Q4_0 GGUF file found in Hugging Face repo {GEMMA_MODEL_REPO!r}; "
                    "set GEMMA_MODEL_FILENAME or GEMMA_MODEL_PATH."
                )
            filename = sorted(q4_files)[0]

        kwargs = {"repo_id": GEMMA_MODEL_REPO, "filename": filename}
        if GEMMA_MODEL_DIR:
            kwargs["local_dir"] = GEMMA_MODEL_DIR
        return hf_hub_download(**kwargs)


def _is_noise_text(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and all(char in "_- .,\n\t" for char in cleaned)


class LocalWhisperASR:
    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None

    def prewarm(self) -> None:
        self._get_model()

    def transcribe(self, pcm_array: np.ndarray) -> str:
        text = self._transcribe(pcm_array)
        if _is_noise_text(text):
            logger.info("Dropping noise-only transcript: %r", text)
            return ""
        return text

    def _transcribe(self, pcm_array: np.ndarray) -> str:
        processor, model, device = self._get_model()
        inputs = processor(
            pcm_array,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        input_features = inputs.input_features.to(device, dtype=model_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generate_kwargs = {"task": WHISPER_TASK}
        if WHISPER_LANGUAGE:
            generate_kwargs["language"] = WHISPER_LANGUAGE

        import torch

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _get_model(self):
        if self._model is not None:
            return self._processor, self._model, self._device

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = torch.device(WHISPER_DEVICE)
        dtype = torch.float16 if WHISPER_DEVICE.startswith("cuda") else torch.float32
        logger.info(
            "Loading Hugging Face Whisper ASR model: %s (device: %s dtype: %s)",
            WHISPER_MODEL,
            device,
            dtype,
        )
        self._processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self._model.eval()
        self._device = device
        return self._processor, self._model, self._device


class LocalGemmaTurnPipeline:
    def __init__(
        self,
        prepare_tts_text,
        interrupt_playback,
        tts: RealtimeOmniVoiceTTS | None = None,
    ):
        self._prepare_tts_text = prepare_tts_text
        self._interrupt_playback = interrupt_playback
        self._asr = LocalWhisperASR()
        self._tts = tts or RealtimeOmniVoiceTTS()
        self._llm = LocalGemmaLLM()

    async def prewarm_tts(self) -> None:
        await self._tts.prewarm()

    async def prewarm_models(self) -> None:
        await asyncio.to_thread(self._asr.prewarm)
        if GEMMA_PREWARM:
            await self._llm.prewarm()
        if REALTIME_TTS_PREWARM:
            await self._tts.prewarm()

    async def run(self, call_id, input_track, output_track, playback_generation):
        if TURN_GREETING_DELAY_SECONDS:
            await asyncio.sleep(TURN_GREETING_DELAY_SECONDS)

        greeting_started_at = time.perf_counter()
        greeting_seconds = await self._speak(
            call_id,
            LOCAL_TURN_GREETING,
            output_track,
            playback_generation,
        )

        logger.info(
            "Greeting timings for %s: tts_wall=%.0f ms tts_audio=%.0f ms",
            call_id,
            (time.perf_counter() - greeting_started_at) * 1000.0,
            greeting_seconds * 1000.0,
        )

        if greeting_seconds:
            protected_seconds = min(
                greeting_seconds + 0.25,
                TURN_GREETING_PROTECTION_MAX_SECONDS,
            )
            logger.info(
                "Protecting Homelands language prompt for %.2f seconds before VAD; audio was %.2f seconds",
                protected_seconds,
                greeting_seconds,
            )
            await self._discard_input_audio(input_track, protected_seconds)

        await self._run_turn_loop(
            call_id=call_id,
            input_track=input_track,
            output_track=output_track,
            playback_generation=playback_generation,
        )

    async def _run_turn_loop(
        self,
        call_id,
        input_track,
        output_track,
        playback_generation,
    ) -> None:
        resampler = AudioResampler(format="s16", layout="mono", rate=16000)
        chunk_buffer = bytearray()
        utterance_buffer = bytearray()
        silence_chunks = 0
        is_speaking = False
        turn_started_at = None

        while True:
            try:
                frame = await input_track.recv()
            except Exception as exc:
                logger.info("Input ended for %s: %s", call_id, exc)
                return

            for resampled in resampler.resample(frame):
                audio_bytes = resampled.to_ndarray().tobytes()
                if not audio_bytes:
                    continue

                chunk_buffer.extend(audio_bytes)

                while len(chunk_buffer) >= TURN_INPUT_CHUNK_SIZE:
                    chunk = bytes(chunk_buffer[:TURN_INPUT_CHUNK_SIZE])
                    del chunk_buffer[:TURN_INPUT_CHUNK_SIZE]

                    audio_np = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))

                    if rms > TURN_SILENCE_THRESHOLD:
                        if not is_speaking:
                            logger.info("Turn VAD: Speech started")
                            is_speaking = True
                            turn_started_at = time.perf_counter()
                            utterance_buffer.clear()
                            silence_chunks = 0
                            self._interrupt_playback(call_id, output_track)

                        utterance_buffer.extend(chunk)
                        silence_chunks = 0

                    elif is_speaking:
                        utterance_buffer.extend(chunk)
                        silence_chunks += 1

                        if silence_chunks >= TURN_END_SILENCE_CHUNKS:
                            logger.info("Turn VAD: Speech ended")
                            is_speaking = False
                            silence_chunks = 0
                            turn_end_at = time.perf_counter()

                            utterance_pcm = bytes(utterance_buffer)
                            utterance_buffer.clear()

                            await self._handle_turn(
                                call_id=call_id,
                                output_track=output_track,
                                playback_generation=playback_generation,
                                turn_started_at=turn_started_at,
                                turn_end_at=turn_end_at,
                                utterance_pcm=utterance_pcm,
                            )

                            turn_started_at = None

    async def _handle_turn(
        self,
        call_id,
        output_track,
        playback_generation,
        turn_started_at: float | None,
        turn_end_at: float,
        utterance_pcm: bytes,
    ) -> None:
        audio_ms = (len(utterance_pcm) / 2) / 16000 * 1000.0
        if audio_ms < TURN_MIN_AUDIO_MS:
            logger.info("Skipping short utterance for %s: %.0f ms", call_id, audio_ms)
            return

        stt_started_at = time.perf_counter()
        transcript_text = await self._transcribe_pcm16(utterance_pcm)
        transcript_ms = (time.perf_counter() - stt_started_at) * 1000.0

        if not transcript_text:
            logger.info("Empty transcript for %s in %.0f ms", call_id, transcript_ms)
            return

        logger.info(
            "Turn transcript for %s in %.0f ms: %s",
            call_id,
            transcript_ms,
            transcript_text,
        )

        llm_started_at = time.perf_counter()
        response_text = await self._generate_response(transcript_text)
        llm_ms = (time.perf_counter() - llm_started_at) * 1000.0

        if not response_text:
            logger.info("Empty Gemma response for %s in %.0f ms", call_id, llm_ms)
            return
        if _is_noise_text(response_text):
            logger.info("Dropping noise-only Gemma response for %s: %r", call_id, response_text)
            return

        logger.info("Turn response for %s in %.0f ms: %s", call_id, llm_ms, response_text)

        tts_started_at = time.perf_counter()
        tts_audio_seconds = await self._speak(
            call_id,
            response_text,
            output_track,
            playback_generation,
        )
        tts_ms = (time.perf_counter() - tts_started_at) * 1000.0

        after_vad_ms = (time.perf_counter() - turn_end_at) * 1000.0
        total_turn_ms = (
            (time.perf_counter() - turn_started_at) * 1000.0
            if turn_started_at is not None
            else after_vad_ms
        )

        logger.info(
            "Turn stages for %s: stt=%.0f ms llm=%.0f ms tts_wall=%.0f ms tts_audio=%.0f ms post_vad=%.0f ms total=%.0f ms",
            call_id,
            transcript_ms,
            llm_ms,
            tts_ms,
            tts_audio_seconds * 1000.0,
            after_vad_ms,
            total_turn_ms,
        )

    async def _transcribe_pcm16(self, pcm: bytes) -> str:
        """Transcribe 16kHz PCM audio using local Whisper model."""
        pcm_array = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: self._asr.transcribe(pcm_array))
        if not text:
            logger.warning("Whisper returned empty transcription")
        return text

    async def _generate_response(self, transcript_text: str) -> str:
        return await self._llm.generate(transcript_text)

    async def _speak(self, call_id, text, output_track, playback_generation):
        prepared = self._prepare_tts_text(text)
        if not prepared:
            return 0.0

        generation_id = playback_generation.get(call_id, 0)
        loop = asyncio.get_running_loop()

        def on_audio_chunk(chunk: bytes, sample_rate: int) -> None:
            if playback_generation.get(call_id, 0) != generation_id:
                return
            loop.call_soon_threadsafe(output_track.add_pcm_audio, chunk, sample_rate)

        audio_seconds = await self._tts.speak(prepared, on_audio_chunk)
        await asyncio.sleep(0)

        if playback_generation.get(call_id, 0) != generation_id:
            logger.info("Stopping interrupted RealtimeTTS playback for %s", call_id)
            return 0.0

        return audio_seconds

    async def _discard_input_audio(self, input_track, duration_seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, duration_seconds)
        discarded_frames = 0

        while time.perf_counter() < deadline:
            try:
                timeout = max(0.01, deadline - time.perf_counter())
                await asyncio.wait_for(input_track.recv(), timeout=timeout)
                discarded_frames += 1
            except asyncio.TimeoutError:
                break
            except Exception as exc:
                logger.info("Protected prompt input drain ended: %s", exc)
                break

        logger.info("Discarded %s inbound frames during protected prompt", discarded_frames)
