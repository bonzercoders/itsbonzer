from __future__ import annotations
import os
import re
import sys
import json
import time
import uuid
import inspect
import queue
import nltk
import torch
import uvicorn
import asyncio
import aiohttp
import logging
import threading
import numpy as np
import multiprocessing
import stream2sentence
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from collections import defaultdict
from collections.abc import Awaitable
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator, Awaitable, Set, Tuple
from server.database import db, Character, Voice, Conversation, MessageCreate
from server.RealtimeSTT import AudioToTextRecorder
from server.stream2sentence import generate_sentences_async
from server.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from server.boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from server.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PLAYBACK_COMPLETE_TIMEOUT_SECONDS = 30.0
PlaybackSessionKey = Tuple[str, str]

########################################
##--          Data Classes          --##
########################################

@dataclass
class CharacterResponse:
    conversation_id: str
    message_id: str
    character_id: str
    character_name: str
    voice_id: str
    text: str = ""

@dataclass
class TTSSentence:
    text: str
    index: int
    turn_id: str
    message_id: str
    character_id: str
    character_name: str
    voice_id: str

@dataclass
class AudioResponseDone:
    """Typed sentinel marking completion of one character response."""
    turn_id: str
    message_id: str
    character_id: str
    character_name: str

@dataclass
class AudioChunk:
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    turn_id: str
    message_id: str
    character_id: str
    character_name: str

@dataclass
class ActiveTextStream:
    turn_id: str
    message_id: str
    character_id: str
    character_name: str
    text: str = ""

@dataclass
class ModelSettings:
    model: str
    temperature: float
    top_p: float
    min_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

@dataclass
class Generation:
    turn_id: str
    last_message: str
    last_responder_id: Optional[str]  # character.id or None for user
    is_user_turn: bool
    responded_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # (responder_id, triggerer_id) â€” who has already responded to whom

    @staticmethod
    def from_user(message: str, turn_id: Optional[str] = None) -> Generation:
        return Generation(
            turn_id=turn_id or str(uuid.uuid4()),
            last_message=message,
            last_responder_id=None,
            is_user_turn=True,
        )

    def after_character(self, message: str, character_id: str) -> Generation:
        """New Generation snapshot after a character responds."""
        new_pairs = set(self.responded_pairs)
        if self.last_responder_id is not None:
            new_pairs.add((character_id, self.last_responder_id))

        return Generation(
            turn_id=self.turn_id,
            last_message=message,
            last_responder_id=character_id,
            is_user_turn=False,
            responded_pairs=new_pairs,
        )

    def can_respond_to_last(self, character_id: str) -> bool:
        """Has this character already responded to whoever spoke last this turn?"""
        if self.last_responder_id is None:
            return True  # anyone can respond to the user
        return (character_id, self.last_responder_id) not in self.responded_pairs

########################################
##--        Queue Management        --##
########################################

class PipeQueues:
    """Queue Management for various pipeline stages"""

    def __init__(self):

        self.stt_queue = asyncio.Queue()
        self.sentence_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()

########################################
##--              STT               --##
########################################

Callback = Callable[..., Optional[Awaitable[None]]]

class STT:
    """Realtime transcription of user's audio prompt"""

    def __init__(self,
                 on_transcription_update: Optional[Callback] = None,
                 on_transcription_stabilized: Optional[Callback] = None,
                 on_transcription_final: Optional[Callback] = None,
                 on_recording_start: Optional[Callback] = None,
                 on_recording_stop: Optional[Callback] = None,
                 on_transcription_start: Optional[Callback] = None,
                 on_vad_detect_start: Optional[Callback] = None,
                 on_vad_detect_stop: Optional[Callback] = None,
                 ):

        # Store callbacks with consistent key names
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_transcription_update': on_transcription_update,
            'on_transcription_stabilized': on_transcription_stabilized,
            'on_transcription_final': on_transcription_final,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
            'on_transcription_start': on_transcription_start,
            'on_vad_detect_start': on_vad_detect_start,
            'on_vad_detect_stop': on_vad_detect_stop,
        }

        self.is_listening = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None

        self.recorder = AudioToTextRecorder(
            model="small.en",
            language="en",
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            realtime_model_type="small.en",
            on_realtime_transcription_update=self._on_transcription_update,
            on_realtime_transcription_stabilized=self._on_transcription_stabilized,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_transcription_start=self._on_transcription_start,
            on_vad_detect_start=self._on_vad_detect_start,
            on_vad_detect_stop=self._on_vad_detect_stop,
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.5,
            spinner=False,
            level=logging.WARNING,
            use_microphone=False
        )

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for callback execution"""

        self.loop = loop

    def transcriber(self):

        while self.is_listening:
            try:
                user_message = self.recorder.text()

                if user_message and user_message.strip():
                    callback = self.callbacks.get('on_transcription_final')
                    if callback:
                        self.run_callback(callback, user_message)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")

    def run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run a user callback from a RealtimeSTT background thread."""

        if callback is None or self.loop is None:
            return

        if inspect.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)
        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""

        if not self.is_listening or not self.recorder:
            return

        try:
            self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
        except Exception as e:
            logger.error(f"Failed to feed audio to recorder: {e}")

    def start_listening(self):
        if self.is_listening:
            return

        self.is_listening = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self.transcriber, daemon=True)
            self._thread.start()

        logger.info("Started listening for audio")

    def stop_listening(self):
        self.is_listening = False
        if self.recorder:
            try:
                self.recorder.abort()
            except Exception as e:
                logger.warning(f"Failed to abort recorder cleanly: {e}")

            try:
                self.recorder.clear_audio_queue()
            except Exception as e:
                logger.warning(f"Failed to clear recorder audio queue: {e}")

        logger.info("Stopped listening for audio")

    def _on_transcription_update(self, text: str) -> None:
        self.run_callback(self.callbacks.get('on_transcription_update'), text)

    def _on_transcription_stabilized(self, text: str) -> None:
        self.run_callback(self.callbacks.get('on_transcription_stabilized'), text)

    def _on_transcription_final(self, user_message: str) -> None:
        self.run_callback(self.callbacks.get('on_transcription_final'), user_message)

    def _on_recording_start(self) -> None:
        self.run_callback(self.callbacks.get('on_recording_start'))

    def _on_recording_stop(self) -> None:
        self.run_callback(self.callbacks.get('on_recording_stop'))

    def _on_transcription_start(self, *_args) -> None:
        self.run_callback(self.callbacks.get('on_transcription_start'))

    def _on_vad_detect_start(self) -> None:
        self.run_callback(self.callbacks.get('on_vad_detect_start'))

    def _on_vad_detect_stop(self) -> None:
        self.run_callback(self.callbacks.get('on_vad_detect_stop'))

########################################
##--              LLM               --##
########################################

class ChatLLM:

    def __init__(self, queues: PipeQueues, api_key: str,
                 on_text_stream_start: Optional[Callable[["Character", str, str], Awaitable[None]]] = None,
                 on_text_stream_stop: Optional[Callable[["Character", str, str, str, bool], Awaitable[None]]] = None,
                 on_text_chunk: Optional[Callable[[str, "Character", str, str], Awaitable[None]]] = None,
                 is_turn_cancelled: Optional[Callable[[str], bool]] = None):

        self.conversation_history: List[Dict] = []
        self.conversation_id: Optional[str] = None
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_settings: Optional[ModelSettings] = None
        self.active_characters: List[Character] = []
        self.user_name: str = "Jay"

        self.on_text_stream_start = on_text_stream_start
        self.on_text_stream_stop = on_text_stream_stop
        self.on_text_chunk = on_text_chunk
        self.is_turn_cancelled = is_turn_cancelled

    async def initialize(self):
        """Load active characters from database on startup."""
        self.active_characters = await db.get_active_characters()
        logger.info(f"ChatLLM initialized with {len(self.active_characters)} active characters")

    async def start_new_conversation(self):
        """Start a new chat session"""
        self.conversation_history = []
        self.conversation_id = str(uuid.uuid4())

    async def clear_conversation_history(self):
        """Clear Conversation History"""
        self.conversation_history = []

    async def get_model_settings(self) -> ModelSettings:
        """Return current model settings, or sensible defaults."""
        if self.model_settings:
            return self.model_settings
        return ModelSettings(
            model="google/gemini-2.5-flash",
            temperature=0.8,
            top_p=0.95,
            min_p=0.05,
            top_k=40,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )

    async def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests"""
        self.model_settings = model_settings

    async def get_active_characters(self) -> List[Character]:
        """Refresh active characters from database"""
        self.active_characters = await db.get_active_characters()
        return self.active_characters

    def character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Explicit Character Instructions â€” pure computation, no I/O."""
        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else.'
        }

    def save_conversation_context(self, messages: List[Dict[str, str]], character: Character, model_settings: ModelSettings) -> str:
        """Save the full conversation context to a JSON file for debugging.

        Returns the filepath of the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"conversation_context_{timestamp}.json"
        filepath = os.path.join("server", filename)

        # Build the full request payload for inspection
        context_data = {
            "timestamp": datetime.now().isoformat(),
            "character": {
                "id": character.id,
                "name": character.name,
            },
            "model_settings": {
                "model": model_settings.model,
                "temperature": model_settings.temperature,
                "top_p": model_settings.top_p,
                "min_p": model_settings.min_p,
                "top_k": model_settings.top_k,
                "frequency_penalty": model_settings.frequency_penalty,
                "presence_penalty": model_settings.presence_penalty,
                "repetition_penalty": model_settings.repetition_penalty,
            },
            "messages": [
                {
                    "role": msg.get("role", ""),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", "")
                }
                for msg in messages
            ],
            "message_count": len(messages),
            "conversation_history_count": len(self.conversation_history),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[DEBUG] Saved conversation context to: {filepath}")
        return filepath

    async def get_user_message(self) -> None:
        """Background task: pull user messages from stt_queue and process."""
        while True:
            try:
                payload = await self.queues.stt_queue.get()
                turn_id: Optional[str] = None
                user_message: str = ""
                if isinstance(payload, tuple) and len(payload) == 2:
                    turn_id, user_message = payload
                elif isinstance(payload, str):
                    user_message = payload
                if user_message and user_message.strip():
                    await self.user_turn(user_message, turn_id=turn_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing user message: {e}")

    async def user_turn(self, user_message: str, turn_id: Optional[str] = None) -> None:
        """Entry point for a new user message. Runs the generation loop
        until no next character is resolved."""

        generation = Generation.from_user(user_message, turn_id=turn_id)

        # append user message exactly once
        self.conversation_history.append({
            "role": "user",
            "name": "Jay",
            "content": user_message,
        })

        while True:
            character = self.determine_next_character(generation)
            if character is None:
                break

            response = await self.initiate_character_response(
                character=character,
                turn_id=generation.turn_id,
                on_text_stream_start=self.on_text_stream_start,
                on_text_stream_stop=self.on_text_stream_stop,
            )

            if not response:
                break

            generation = generation.after_character(response, character.id)

    def parse_last_message(
        self,
        text: str,
        active_characters: List[Character],
        exclude_id: Optional[str] = None,
    ) -> Optional[Character]:
        """Find the first active character mentioned in text.

        Matches full name, first name, or last name using word boundaries.
        Skips the character who just spoke (exclude_id).
        """
        text_lower = text.lower()

        for character in active_characters:
            if exclude_id and character.id == exclude_id:
                continue

            name_parts = character.name.lower().split()

            # check full name first, then individual parts
            patterns = [re.escape(character.name.lower())]
            patterns.extend(re.escape(part) for part in name_parts)

            for pattern in patterns:
                if re.search(rf"\b{pattern}\b", text_lower):
                    return character

        return None

    def determine_next_character(self, generation: Generation) -> Optional[Character]:
        """Decide who speaks next.

        1. Parse last message for a character mention.
        2. Check loop deterrent â€” has this pair already fired this turn?
        3. User turn with no mention â†’ default to first character.
        4. Character turn with no mention â†’ cycle ends.
        """
        mentioned = self.parse_last_message(
            text=generation.last_message,
            active_characters=self.active_characters,
            exclude_id=generation.last_responder_id,
        )

        if mentioned:
            if generation.can_respond_to_last(mentioned.id):
                return mentioned
            return None  # blocked by loop deterrent

        # user spoke but didn't mention anyone  default character
        if generation.is_user_turn and self.active_characters:
            return self.active_characters[0]

        return None

    def build_character_messages(self, character: Character) -> List[Dict[str, str]]:
        """Build the full message list for an OpenRouter request. Pure computation."""

        messages = []

        if character.global_roleplay:
            messages.append({"role": "system", "content": character.global_roleplay})

        if character.system_prompt:
            messages.append({"role": "system", "content": character.system_prompt})

        messages.extend(self.conversation_history)

        messages.append(self.character_instruction_message(character))

        return messages

    async def initiate_character_response(self,
                                          character: Character,
                                          turn_id: str,
                                          on_text_stream_start: Optional[Callable[[Character, str, str], Awaitable[None]]] = None,
                                          on_text_stream_stop: Optional[Callable[[Character, str, str, str, bool], Awaitable[None]]] = None) -> Optional[str]:

        if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
            return None

        model_settings = await self.get_model_settings()
        message_id = str(uuid.uuid4())
        messages = self.build_character_messages(character)

        if self.on_text_stream_start:
            await self.on_text_stream_start(character, message_id, turn_id)

        response = await self.stream_character_response(messages=messages,
                                                        character=character,
                                                        message_id=message_id,
                                                        turn_id=turn_id,
                                                        model_settings=model_settings,
                                                        on_text_chunk=self.on_text_chunk)

        interrupted = bool(self.is_turn_cancelled and self.is_turn_cancelled(turn_id))
        if self.on_text_stream_stop:
            await self.on_text_stream_stop(character, message_id, response, turn_id, interrupted)

        if response and not interrupted:
            self.conversation_history.append({"role": "assistant","name": character.name,"content": response})
            return response

        return None

    async def stream_character_response(self,
                                        messages: List[Dict[str, str]],
                                        character: Character,
                                        message_id: str,
                                        turn_id: str,
                                        model_settings: ModelSettings,
                                        on_text_chunk: Optional[Callable[[str, Character, str, str], Awaitable[None]]] = None) -> str:
        """Stream LLM tokens, split into sentences, push TTSSentence items to sentence_queue."""

        sentence_index = 0
        response = ""

        if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
            return response

        self.save_conversation_context(messages, character, model_settings)

        try:
            stream = await self.client.chat.completions.create(
                model=model_settings.model,
                messages=messages,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
                stream=True,
                extra_body={
                    "top_k": model_settings.top_k,
                    "min_p": model_settings.min_p,
                    "repetition_penalty": model_settings.repetition_penalty,
                }
            )

            async def chunk_generator() -> AsyncGenerator[str, None]:
                nonlocal response
                async for chunk in stream:
                    if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
                        break
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content
                        if content:
                            response += content
                            if on_text_chunk:
                                await on_text_chunk(content, character, message_id, turn_id)
                            yield content

            async for sentence in generate_sentences_async(
                chunk_generator(),
                minimum_first_fragment_length=14,
                minimum_sentence_length=25,
                tokenizer="nltk",
                quick_yield_single_sentence_fragment=True,
                sentence_fragment_delimiters=".?!;:,\nâ€¦)]}ã€‚-",
                full_sentence_delimiters=".?!\nâ€¦ã€‚",
            ):
                sentence_text = sentence.strip()
                if sentence_text:
                    if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
                        break
                    await self.queues.sentence_queue.put(TTSSentence(
                        text=sentence_text,
                        index=sentence_index,
                        turn_id=turn_id,
                        message_id=message_id,
                        character_id=character.id,
                        character_name=character.name,
                        voice_id=character.voice_id,
                    ))
                    logger.info(f"[LLM] {character.name} sentence {sentence_index}: {sentence_text[:50]}...")
                    sentence_index += 1

        except Exception as e:
            logger.error(f"[LLM] Error streaming for {character.name}: {e}")

        finally:
            await self.queues.sentence_queue.put(AudioResponseDone(
                turn_id=turn_id,
                message_id=message_id,
                character_id=character.id,
                character_name=character.name,
            ))
            logger.info(f"[LLM] {character.name} complete: {sentence_index} sentences, sentinel enqueued")

        return response

########################################
##--         Text to Speech         --##
########################################

def revert_delay_pattern(data: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Undo Higgs delay pattern so decoded frames line up."""
    if data.ndim != 2:
        raise ValueError('Expected 2D tensor from audio tokenizer')
    if data.shape[1] - data.shape[0] < start_idx:
        raise ValueError('Invalid start_idx for delay pattern reversion')

    out = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out.append(data[i:(i + 1), i + start_idx:(data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out, dim=0)

class TTS:

    def __init__(self, queues: PipeQueues, is_turn_cancelled: Optional[Callable[[str], bool]] = None):
        self.queues = queues
        self._task_tts_worker: Optional[asyncio.Task] = None
        self.is_turn_cancelled = is_turn_cancelled

        # Set during initialize()
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.sample_rate: int = 24000
        self.chunk_size: int = 14
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.voice_dir = os.path.join(os.path.dirname(__file__), "voices")

    async def initialize(self):
        """Initialize the Higgs Audio engine. Called once at startup."""
        try:

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.engine = HiggsAudioServeEngine(
                model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
                audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
                device=device
            )

        except Exception as e:
            logger.error(f"Failed to initialize Higgs Audio TTS: {e}")
            raise

    async def tts_worker(self):
        """Background task: pull sentences from queue, synthesize audio, queue chunks."""

        while True:
            try:
                item = await self.queues.sentence_queue.get()

            except asyncio.CancelledError:
                break

            try:
                if isinstance(item, AudioResponseDone):
                    if self.is_turn_cancelled and self.is_turn_cancelled(item.turn_id):
                        logger.info(f"[TTS] Dropping completion sentinel for cancelled turn {item.turn_id}")
                        continue
                    await self.queues.tts_queue.put(item)
                    logger.info(f"[TTS] End of response for {item.character_name}")
                    continue

                sentence: TTSSentence = item
                if self.is_turn_cancelled and self.is_turn_cancelled(sentence.turn_id):
                    logger.info(f"[TTS] Dropping sentence {sentence.index} for cancelled turn {sentence.turn_id}")
                    continue

                logger.info(f"[TTS] Generating audio for sentence {sentence.index}")
                chunk_index = 0

                try:
                    async for pcm_bytes in self.synthesize_speech(sentence.text, sentence.voice_id, sentence.turn_id):
                        if self.is_turn_cancelled and self.is_turn_cancelled(sentence.turn_id):
                            logger.info(f"[TTS] Stopping synthesis for cancelled turn {sentence.turn_id}")
                            break
                        await self.queues.tts_queue.put(AudioChunk(audio_bytes=pcm_bytes,
                                                                   sentence_index=sentence.index,
                                                                   chunk_index=chunk_index,
                                                                   turn_id=sentence.turn_id,
                                                                   message_id=sentence.message_id,
                                                                   character_id=sentence.character_id,
                                                                   character_name=sentence.character_name)
                                                                   )
                        chunk_index += 1

                    logger.info(f"[TTS] {sentence.character_name} #{sentence.index}: {chunk_index} chunks")

                except Exception as e:
                    logger.error(f"[TTS] Error generating audio: {e}")

            finally:
                self.queues.sentence_queue.task_done()

    def _resolve_reference_path(self, path_value: str) -> str:
        """Resolve absolute/relative reference paths for voice assets."""
        candidate = (path_value or "").strip()
        if not candidate:
            raise ValueError("Voice reference path is empty")

        if os.path.exists(candidate):
            return candidate

        in_voice_dir = os.path.join(self.voice_dir, candidate)
        if os.path.exists(in_voice_dir):
            return in_voice_dir

        raise FileNotFoundError(f"Voice reference path not found: {path_value}")

    async def load_voice_reference(self, voice: Voice):
        """Load reference audio and text for voice cloning from a Voice record."""
        if not voice.ref_audio:
            raise ValueError(f"Voice '{voice.voice_id}' is missing required ref_audio")

        audio_path = self._resolve_reference_path(voice.ref_audio)
        ref_text_value = (voice.ref_text or "").strip()
        if not ref_text_value:
            raise ValueError(f"Voice '{voice.voice_id}' is missing required ref_text")

        if os.path.exists(ref_text_value):
            with open(ref_text_value, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
        else:
            ref_text = ref_text_value

        if not ref_text:
            raise ValueError(f"Voice '{voice.voice_id}' resolved to empty reference text")

        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

        return messages
    
    async def synthesize_speech(self, text: str, voice_id: str, turn_id: str) -> AsyncGenerator[bytes, None]:
        """Stream PCM16 audio chunks from Higgs Audio engine."""

        if not voice_id:
            raise ValueError("Cannot synthesize speech without voice_id")
        if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
            return

        selected_voice = await db.get_voice(voice_id)
        messages = await self.load_voice_reference(selected_voice)
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop_strings=['<|end_of_text|>', '<|eot_id|>'],
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                force_audio_gen=True,
            ):
                if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
                    break

                if delta.audio_tokens is None:
                    continue

                # Check for end token (1025)
                if torch.all(delta.audio_tokens == 1025):
                    break

                # Accumulate tokens
                audio_tokens.append(delta.audio_tokens[:, None])

                # Count non-padding tokens (1024 is padding)
                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1

                # Decode when chunk size reached
                if seq_len > 0 and seq_len % self.chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    try:
                        # Revert delay pattern and decode
                        vq_code = (revert_delay_pattern(audio_tensor, start_idx=seq_len - self.chunk_size + 1).clip(0, 1023).to(self.device))
                        waveform = self.engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                        # Convert to numpy
                        if isinstance(waveform, torch.Tensor):
                            waveform_np = waveform.detach().cpu().numpy()
                        else:
                            waveform_np = np.asarray(waveform, dtype=np.float32)

                        # Convert to PCM16 bytes
                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()

                    except Exception as e:
                        logger.warning(f"Error decoding chunk: {e}")
                        continue

        # Flush remaining tokens
        if seq_len > 0 and seq_len % self.chunk_size != 0 and audio_tokens:
            if self.is_turn_cancelled and self.is_turn_cancelled(turn_id):
                return
            audio_tensor = torch.cat(audio_tokens, dim=-1)
            remaining = seq_len % self.chunk_size

            try:
                vq_code = (revert_delay_pattern(audio_tensor, start_idx=seq_len - remaining + 1).clip(0, 1023).to(self.device))
                waveform = self.engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.detach().cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform, dtype=np.float32)

                pcm = np.clip(waveform_np, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

            except Exception as e:
                logger.warning(f"Error flushing remaining audio: {e}")

    async def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available voices formatted for frontend."""
        try:
            db_voices = await db.get_all_voices()
            voices = [
                {"voice_id": voice.voice_id, "voice_name": voice.voice_name}
                for voice in db_voices
            ]
            voices.sort(key=lambda item: item["voice_name"].lower())
            return voices
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []

    def shutdown(self):
        """Cleanup resources"""
        logger.info('Shutting down TTS manager')
        self.engine = None
        self._initialized = False

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connection and routes messages."""

    def __init__(self):
        self.queues = PipeQueues()
        self.websocket: Optional[WebSocket] = None
        self.stt: Optional[STT] = None
        self.chat: Optional[ChatLLM] = None
        self.tts: Optional[TTS] = None

        self._task_user_message: Optional[asyncio.Task] = None
        self._task_tts_worker: Optional[asyncio.Task] = None
        self._task_stream_audio: Optional[asyncio.Task] = None

        self.user_name = "Jay"
        self.stream_start_time: Optional[float] = None
        self.active_turn_id: Optional[str] = None
        self.cancelled_turn_ids: Set[str] = set()
        self.tts_is_playing = False
        self.active_playback_sessions: Set[PlaybackSessionKey] = set()
        self.playback_stop_timeouts: Dict[PlaybackSessionKey, asyncio.TimerHandle] = {}
        self.active_audio_stream: Optional[AudioChunk] = None
        self.active_text_stream: Optional[ActiveTextStream] = None
        self.stt_state: str = "inactive"

    async def initialize(self):
        """Initialize all pipeline components at startup."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.stt = STT(
            on_transcription_update=self.on_transcription_update,
            on_transcription_stabilized=self.on_transcription_stabilized,
            on_transcription_final=self.on_transcription_final,
            on_recording_start=self.on_recording_start,
            on_recording_stop=self.on_recording_stop,
            on_transcription_start=self.on_transcription_start,
            on_vad_detect_start=self.on_vad_detect_start,
            on_vad_detect_stop=self.on_vad_detect_stop,
        )

        self.stt.set_event_loop(asyncio.get_event_loop())

        self.chat = ChatLLM(
            queues=self.queues,
            api_key=api_key,
            on_text_stream_start=self.on_text_stream_start,
            on_text_stream_stop=self.on_text_stream_stop,
            on_text_chunk=self.on_text_chunk,
            is_turn_cancelled=self.is_turn_cancelled,
        )
        await self.chat.initialize()

        self.tts = TTS(queues=self.queues, is_turn_cancelled=self.is_turn_cancelled)
        await self.tts.initialize()

        logger.info(f"Initialized with {len(self.chat.active_characters)} active characters")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection, start pipeline and audio streamer."""
        await websocket.accept()
        self.websocket = websocket

        await self._conversation_tasks()
        await self.emit_stt_state()

        logger.info("WebSocket connected")

    async def disconnect(self):
        """Stop everything cleanly on WebSocket close."""
        if self.stt:
            self.stt.stop_listening()

        self.websocket = None
        self.stt_state = "inactive"
        self._clear_playback_sessions(reason="disconnect")
        self.active_audio_stream = None
        self.active_text_stream = None

    async def shutdown(self):
        await self.disconnect()

    async def _conversation_tasks(self) -> None:
        if self.chat and (self._task_user_message is None or self._task_user_message.done()):
            self._task_user_message = asyncio.create_task(self.chat.get_user_message())

        if self.tts and (self._task_tts_worker is None or self._task_tts_worker.done()):
            self._task_tts_worker = asyncio.create_task(self.tts.tts_worker())

        if self._task_stream_audio is None or self._task_stream_audio.done():
            self._task_stream_audio = asyncio.create_task(self.stream_audio())

    def is_turn_cancelled(self, turn_id: str) -> bool:
        return turn_id in self.cancelled_turn_ids

    async def emit_stt_state(self) -> None:
        await self.send_text_to_client({"type": "stt_state", "data": {"state": self.stt_state}})

    async def set_stt_state(self, next_state: str) -> None:
        if self.stt_state == next_state:
            return
        self.stt_state = next_state
        await self.emit_stt_state()

    def _set_tts_playback_state(self, is_playing: bool, reason: str) -> None:
        if self.tts_is_playing == is_playing:
            return
        self.tts_is_playing = is_playing
        logger.info(f"[TTS] tts_is_playing={is_playing} ({reason})")

    def _playback_session_key(self, character_id: str, message_id: str) -> PlaybackSessionKey:
        return (character_id, message_id)

    def _cancel_playback_timeout(self, session_key: PlaybackSessionKey) -> None:
        timeout_handle = self.playback_stop_timeouts.pop(session_key, None)
        if timeout_handle:
            timeout_handle.cancel()

    def _refresh_tts_playing_from_sessions(self, reason: str) -> None:
        self._set_tts_playback_state(len(self.active_playback_sessions) > 0, reason)

    def _mark_playback_started(self, character_id: str, message_id: str) -> None:
        session_key = self._playback_session_key(character_id, message_id)
        self._cancel_playback_timeout(session_key)
        self.active_playback_sessions.add(session_key)
        logger.info(
            f"[TTS] Playback started for {character_id}/{message_id}; "
            f"active_sessions={len(self.active_playback_sessions)}"
        )
        self._refresh_tts_playing_from_sessions("audio_stream_start")

    def _schedule_playback_stop_failsafe(self, character_id: str, message_id: str) -> None:
        session_key = self._playback_session_key(character_id, message_id)
        if session_key not in self.active_playback_sessions:
            return

        self._cancel_playback_timeout(session_key)

        def expire() -> None:
            self.playback_stop_timeouts.pop(session_key, None)
            if session_key not in self.active_playback_sessions:
                return

            self.active_playback_sessions.discard(session_key)
            logger.warning(
                f"[TTS] Playback completion failsafe expired for "
                f"{character_id}/{message_id}; active_sessions={len(self.active_playback_sessions)}"
            )
            self._refresh_tts_playing_from_sessions("playback_failsafe_expired")

        loop = asyncio.get_running_loop()
        self.playback_stop_timeouts[session_key] = loop.call_later(
            PLAYBACK_COMPLETE_TIMEOUT_SECONDS,
            expire,
        )

    def _clear_playback_sessions(self, reason: str) -> None:
        active_session_count = len(self.active_playback_sessions)

        for timeout_handle in self.playback_stop_timeouts.values():
            timeout_handle.cancel()
        self.playback_stop_timeouts.clear()
        self.active_playback_sessions.clear()

        if active_session_count > 0:
            logger.info(f"[TTS] Cleared {active_session_count} playback session(s) ({reason})")
        self._set_tts_playback_state(False, reason)

    async def on_audio_playback_complete(self, character_id: str, message_id: str) -> None:
        session_key = self._playback_session_key(character_id, message_id)
        self._cancel_playback_timeout(session_key)

        did_remove = session_key in self.active_playback_sessions
        self.active_playback_sessions.discard(session_key)
        logger.info(
            f"[TTS] Client playback complete for {character_id}/{message_id}; "
            f"known_session={did_remove}; active_sessions={len(self.active_playback_sessions)}"
        )
        self._refresh_tts_playing_from_sessions("audio_playback_complete")

    async def start_new_turn(self, turn_id: str) -> None:
        if self.active_turn_id and self.active_turn_id != turn_id:
            await self.cancel_active_turn(reason="new_turn", emit_client_event=False)
        self.active_turn_id = turn_id
        self.cancelled_turn_ids.discard(turn_id)

    def _drain_queue_for_cancelled_turns(self, target_queue: asyncio.Queue) -> int:
        retained: List[Any] = []
        removed = 0

        while True:
            try:
                item = target_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            target_queue.task_done()
            item_turn_id = getattr(item, "turn_id", None)
            if item_turn_id in self.cancelled_turn_ids:
                removed += 1
                continue
            retained.append(item)

        for item in retained:
            target_queue.put_nowait(item)

        return removed

    async def cancel_active_turn(self, reason: str, emit_client_event: bool = True) -> Optional[str]:
        turn_id = self.active_turn_id
        if not turn_id:
            return None
        if turn_id in self.cancelled_turn_ids:
            return turn_id

        self.cancelled_turn_ids.add(turn_id)

        interrupted_message_id: Optional[str] = None
        if self.active_text_stream and self.active_text_stream.turn_id == turn_id:
            interrupted_message_id = self.active_text_stream.message_id

        if emit_client_event and reason == "barge_in":
            interruption_data: Dict[str, str] = {"reason": "barge_in"}
            if interrupted_message_id:
                interruption_data["message_id"] = interrupted_message_id
            await self.send_text_to_client({
                "type": "tts_interrupted",
                "data": interruption_data,
            })

        if self.active_audio_stream and self.active_audio_stream.turn_id == turn_id:
            await self.on_audio_stream_stop(self.active_audio_stream)
            self.active_audio_stream = None

        if self.active_text_stream and self.active_text_stream.turn_id == turn_id:
            interrupted_stream = self.active_text_stream
            self.active_text_stream = None
            await self.send_text_to_client({
                "type": "text_stream_stop",
                "data": {
                    "character_id": interrupted_stream.character_id,
                    "character_name": interrupted_stream.character_name,
                    "message_id": interrupted_stream.message_id,
                    "text": interrupted_stream.text,
                    "interrupted": True,
                },
            })

        removed_sentences = self._drain_queue_for_cancelled_turns(self.queues.sentence_queue)
        removed_audio = self._drain_queue_for_cancelled_turns(self.queues.tts_queue)

        logger.info(
            f"[TURN] Cancelled {turn_id} ({reason}); drained {removed_sentences} sentence items and {removed_audio} audio items"
        )

        self._clear_playback_sessions(reason=f"cancel_active_turn:{reason}")
        if self.active_turn_id == turn_id:
            self.active_turn_id = None

        if len(self.cancelled_turn_ids) > 1024:
            self.cancelled_turn_ids.clear()

        return turn_id

    async def stream_audio(self) -> None:
        """Long-running consumer: pull audio chunks from tts_queue and stream to client."""
        try:
            while True:
                item = await self.queues.tts_queue.get()
                try:
                    if isinstance(item, AudioResponseDone):
                        if self.is_turn_cancelled(item.turn_id):
                            continue

                        if (
                            self.active_audio_stream
                            and self.active_audio_stream.turn_id == item.turn_id
                            and self.active_audio_stream.message_id == item.message_id
                        ):
                            await self.on_audio_stream_stop(self.active_audio_stream)
                            self.active_audio_stream = None
                        continue

                    chunk: AudioChunk = item

                    if self.is_turn_cancelled(chunk.turn_id):
                        continue
                    if self.active_turn_id and chunk.turn_id != self.active_turn_id:
                        continue

                    if (
                        self.active_audio_stream is None
                        or self.active_audio_stream.turn_id != chunk.turn_id
                        or self.active_audio_stream.message_id != chunk.message_id
                    ):
                        await self.on_audio_stream_start(chunk)

                    self.active_audio_stream = chunk

                    if self.websocket:
                        await self.websocket.send_bytes(chunk.audio_bytes)
                finally:
                    self.queues.tts_queue.task_done()

        except asyncio.CancelledError:
            logger.debug("[Transport] Audio streamer cancelled")

    async def refresh_active_characters(self):
        """Refresh active characters from database (call when characters change)."""
        if self.chat:
            self.chat.active_characters = await self.chat.get_active_characters()
            logger.info(f"Refreshed to {len(self.chat.active_characters)} active characters")

    # ------ Pipeline event callbacks ------ #

    async def on_transcription_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})

    async def on_transcription_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})

    async def on_recording_start(self):
        await self.set_stt_state("recording")
        if self.tts_is_playing:
            await self.cancel_active_turn(reason="barge_in", emit_client_event=True)

    async def on_recording_stop(self):
        if self.stt and self.stt.is_listening:
            await self.set_stt_state("listening")

    async def on_transcription_start(self):
        await self.set_stt_state("transcribing")

    async def on_vad_detect_start(self):
        if self.stt and self.stt.is_listening and self.stt_state == "inactive":
            await self.set_stt_state("listening")

    async def on_vad_detect_stop(self):
        if self.stt and self.stt.is_listening and self.stt_state != "inactive":
            await self.set_stt_state("listening")

    async def on_transcription_final(self, user_message: str):
        turn_id = str(uuid.uuid4())
        await self.start_new_turn(turn_id)
        self.stream_start_time = time.time()
        await self.queues.stt_queue.put((turn_id, user_message))
        await self.send_text_to_client({"type": "stt_final", "text": user_message})
        await self.set_stt_state("listening")

    async def on_text_stream_start(self, character: Character, message_id: str, turn_id: str):
        if self.is_turn_cancelled(turn_id):
            return

        self.active_text_stream = ActiveTextStream(
            turn_id=turn_id,
            message_id=message_id,
            character_id=character.id,
            character_name=character.name,
            text="",
        )

        await self.send_text_to_client({
            "type": "text_stream_start",
            "data": {
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
            },
        })

    async def on_text_chunk(self, text: str, character: Character, message_id: str, turn_id: str):
        """Forward a single LLM text chunk to the client."""
        if self.is_turn_cancelled(turn_id):
            return

        if (
            self.active_text_stream
            and self.active_text_stream.turn_id == turn_id
            and self.active_text_stream.message_id == message_id
        ):
            self.active_text_stream.text += text

        await self.send_text_to_client({
            "type": "text_chunk",
            "data": {
                "text": text,
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
            },
        })

    async def on_text_stream_stop(self, character: Character, message_id: str, text: str, turn_id: str, interrupted: bool):
        if self.is_turn_cancelled(turn_id) and not interrupted:
            interrupted = True

        if (
            self.active_text_stream
            and self.active_text_stream.turn_id == turn_id
            and self.active_text_stream.message_id == message_id
        ):
            if not text and self.active_text_stream.text:
                text = self.active_text_stream.text
            self.active_text_stream = None
        elif self.is_turn_cancelled(turn_id):
            return

        await self.send_text_to_client({
            "type": "text_stream_stop",
            "data": {
                "character_id": character.id,
                "character_name": character.name,
                "message_id": message_id,
                "text": text,
                "interrupted": interrupted,
            },
        })

    async def on_audio_stream_start(self, chunk: AudioChunk):
        self._mark_playback_started(chunk.character_id, chunk.message_id)
        sample_rate = self.tts.sample_rate if self.tts else 24000
        await self.send_text_to_client({
            "type": "audio_stream_start",
            "data": {
                "character_id": chunk.character_id,
                "character_name": chunk.character_name,
                "message_id": chunk.message_id,
                "sample_rate": sample_rate,
            },
        })

        if self.stream_start_time is not None:
            latency = time.time() - self.stream_start_time
            logger.info(f"Audio stream start, latency to first chunk: {latency:.2f}s")

    async def on_audio_stream_stop(self, chunk: AudioChunk):
        logger.info(
            f"[TTS] Emitting audio_stream_stop for {chunk.character_id}/{chunk.message_id}"
        )
        await self.send_text_to_client({
            "type": "audio_stream_stop",
            "data": {
                "character_id": chunk.character_id,
                "character_name": chunk.character_name,
                "message_id": chunk.message_id,
            },
        })
        self._schedule_playback_stop_failsafe(chunk.character_id, chunk.message_id)

    # ------ WebSocket message handling ------ #

    async def handle_text_message(self, raw: str):
        """Parse incoming JSON text message and route to handler."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON text message: {raw[:100]}")
            return

        message_type = data.get("type", "")

        if message_type == "ping":
            await self.send_text_to_client({"type": "pong"})

        elif message_type == "user_message":
            text = data.get("text", "").strip()
            if text:
                await self.handle_user_message(text)

        elif message_type == "start_listening":
            if self.stt:
                self.stt.start_listening()
                await self.set_stt_state("listening")

        elif message_type == "stop_listening":
            if self.stt:
                self.stt.stop_listening()
                await self.set_stt_state("inactive")

        elif message_type == "audio_playback_complete":
            character_id = data.get("character_id")
            message_id = data.get("message_id")
            if isinstance(character_id, str) and isinstance(message_id, str):
                await self.on_audio_playback_complete(character_id, message_id)
            else:
                logger.warning(
                    "[TTS] Received malformed audio_playback_complete payload: "
                    f"{data}"
                )

        elif message_type == "model_settings":
            model_settings = ModelSettings(
                model=data.get("model", "openai/gpt-oss-120b"),
                temperature=float(data.get("temperature", 0.93)),
                top_p=float(data.get("top_p", 0.95)),
                min_p=float(data.get("min_p", 0.0)),
                top_k=int(data.get("top_k", 40)),
                frequency_penalty=float(data.get("frequency_penalty", 0.0)),
                presence_penalty=float(data.get("presence_penalty", 0.0)),
                repetition_penalty=float(data.get("repetition_penalty", 1.0)),
            )
            if self.chat:
                await self.chat.set_model_settings(model_settings)
            logger.info(f"Model settings updated: {model_settings.model}")

        elif message_type == "clear_history":
            if self.chat:
                await self.chat.clear_conversation_history()

        elif message_type == "refresh_characters":
            await self.refresh_active_characters()

        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def handle_audio_message(self, audio_bytes: bytes):
        """Feed audio for transcription."""
        if self.stt:
            self.stt.feed_audio(audio_bytes)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message (typed, not from STT)."""
        turn_id = str(uuid.uuid4())
        await self.start_new_turn(turn_id)
        self.stream_start_time = time.time()
        await self.queues.stt_queue.put((turn_id, user_message))

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client."""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))
########################################
##--           FastAPI App          --##
########################################

ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up services...")

    await db.init_database()
    await ws_manager.initialize()

    db.on_characters_changed = ws_manager.refresh_active_characters
    await db.subscribe_to_broadcasts()

    print("All services initialised!")

    yield

    print("Shutting down services...")
    await ws_manager.shutdown()
    print("All services shut down!")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
##--       WebSocket Endpoint       --##
########################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                await ws_manager.handle_text_message(message["text"])

            elif "bytes" in message:
                await ws_manager.handle_audio_message(message["bytes"])

    except WebSocketDisconnect:
        await ws_manager.disconnect()

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect()

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="client", html=True), name="client")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


