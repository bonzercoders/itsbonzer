"""
Database Module
Centralized database operations for Characters, Voices, Conversations, and Messages
using Supabase as the backend.
"""

import os
import re
import json
import asyncio
import logging
import threading
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
from pydantic import BaseModel
from supabase import acreate_client, AsyncClient
from fastapi import HTTPException

logger = logging.getLogger(__name__)

########################################
##--         Configuration          --##
########################################

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wnqozfubwnwqksjkksfy.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_T28Uvx3MVYHo0opH0Eweqw_kCCNFuvC")

########################################
##--          Data Models           --##
########################################

# Character Models
class Character(BaseModel):
    id: str
    name: str
    voice_id: str = ""
    global_roleplay: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CharacterCreate(BaseModel):
    name: str
    voice_id: str = ""
    global_roleplay: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    voice_id: Optional[str] = None
    global_roleplay: Optional[str] = None
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None


# Voice Models
class Voice(BaseModel):
    voice_id: str           # Primary key
    voice_name: str         # Display name (human-readable)
    method: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""
    audio_ids: Optional[Any] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VoiceCreate(BaseModel):
    voice_id: str
    voice_name: str
    method: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""


class VoiceUpdate(BaseModel):
    voice_name: Optional[str] = None
    method: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    speaker_desc: Optional[str] = None
    scene_prompt: Optional[str] = None
    audio_ids: Optional[Any] = None


# Conversation Models
class Conversation(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    active_characters: List[Dict[str, Any]] = []


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    active_characters: Optional[List[Dict[str, Any]]] = None


# Message Models
class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    name: Optional[str] = None
    content: str
    character_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    name: Optional[str] = None
    character_id: Optional[str] = None


########################################
##--            Database            --##
########################################

class Database:
    """
    Centralized database management for all Supabase operations.
    Handles Characters, Voices, Conversations, and Messages.
    """

    def __init__(self, supabase_client: Optional[AsyncClient] = None):
        """Initialize with optional async Supabase client."""
        self.supabase = supabase_client

        # Voice cache for performance
        self._voice_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

        # Broadcast channels
        self._character_channel = None
        self._voice_channel = None

        # Callbacks for broadcast events
        self.on_characters_changed: Optional[Callable[[], Awaitable[None]]] = None
        self.on_voices_changed: Optional[Callable[[], Awaitable[None]]] = None

    async def init_database(self):
        """Initialize async client and validate database connection on startup."""
        try:
            if self.supabase is None:
                self.supabase = await acreate_client(SUPABASE_URL, SUPABASE_KEY)

            # Quick connectivity check
            await self.supabase.table("characters").select("id").limit(1).execute()
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            raise

    async def subscribe_to_broadcasts(self):
        """
        Subscribe to Realtime Broadcast channels for character and voice changes.
        Frontend broadcasts when it creates/updates/deletes characters or voices.
        """
        if self.supabase is None:
            raise RuntimeError("Supabase client not initialized. Call init_database() first.")

        # Prevent duplicate subscriptions on hot-reload/restart paths.
        await self.shutdown_realtime()

        # Character changes channel
        self._character_channel = self.supabase.channel('db-characters')
        self._character_channel.on_broadcast('character-changed', self._on_character_changed)
        await self._character_channel.subscribe()
        logger.info("Subscribed to db-characters broadcast channel")

        # Voice changes channel
        self._voice_channel = self.supabase.channel('db-voices')
        self._voice_channel.on_broadcast('voice-changed', self._on_voice_changed)
        await self._voice_channel.subscribe()
        logger.info("Subscribed to db-voices broadcast channel")

    async def shutdown_realtime(self):
        """Unsubscribe/remove realtime channels during shutdown or re-init."""
        if self.supabase is None:
            return

        for channel_attr in ("_character_channel", "_voice_channel"):
            channel = getattr(self, channel_attr)
            if channel is None:
                continue
            try:
                await self.supabase.remove_channel(channel)
            except Exception as e:
                logger.warning(f"Failed removing channel {channel_attr}: {e}")
            finally:
                setattr(self, channel_attr, None)

    @staticmethod
    def _schedule_callback(callback: Optional[Callable[[], Awaitable[None]]]) -> None:
        if callback is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop; dropping Supabase callback")
            return

        try:
            coro = callback()
        except Exception as e:
            logger.error(f"Supabase callback failed before scheduling: {e}")
            return

        task = loop.create_task(coro)
        task.add_done_callback(Database._log_task_exception)

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Supabase callback failed: {exc}")

    def _on_character_changed(self, payload):
        """Callback: frontend changed a character."""
        logger.info(f"Broadcast received: character-changed {payload}")
        self._schedule_callback(self.on_characters_changed)

    def _on_voice_changed(self, payload):
        """Callback: frontend changed a voice. Clear the cache."""
        logger.info(f"Broadcast received: voice-changed {payload}")
        self.clear_voice_cache()
        self._schedule_callback(self.on_voices_changed)

    ########################################
    ##--      Character Operations      --##
    ########################################

    async def _generate_character_id(self, name: str) -> str:
        """Generate a sequential ID from the character name."""
        base_id = name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)
        base_id = re.sub(r'\s+', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        base_id = base_id.strip('-')

        try:
            response = await self.supabase.table("characters")\
                .select("id")\
                .like("id", f"{base_id}-%")\
                .execute()

            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")

            for row in response.data:
                match = pattern.match(row["id"])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)

            next_num = highest_num + 1
            character_id = f"{base_id}-{next_num:03d}"

            logger.info(f"Generated character id: {character_id}")
            return character_id

        except Exception as e:
            logger.error(f"Error generating character id: {e}")
            return f"{base_id}-001"

    async def get_all_characters(self) -> List[Character]:
        """Get all characters."""
        try:
            response = await self.supabase.table("characters")\
                .select("*")\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice_id": row.get("voice_id") or "",
                    "global_roleplay": row.get("global_roleplay") or "",
                    "system_prompt": row.get("system_prompt") or "",
                    "image_url": row.get("image_url") or "",
                    "images": row.get("images") or [],
                    "is_active": row.get("is_active") or False,
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
                characters.append(Character(**character_data))

            logger.info(f"Retrieved {len(characters)} characters")
            return characters

        except Exception as e:
            logger.error(f"Error getting characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_active_characters(self) -> List[Character]:
        """Get all active characters."""
        try:
            response = await self.supabase.table("characters")\
                .select("*")\
                .eq("is_active", True)\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice_id": row.get("voice_id") or "",
                    "global_roleplay": row.get("global_roleplay") or "",
                    "system_prompt": row.get("system_prompt") or "",
                    "image_url": row.get("image_url") or "",
                    "images": row.get("images") or [],
                    "is_active": row.get("is_active") or False,
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
                characters.append(Character(**character_data))

            logger.info(f"Retrieved {len(characters)} active characters")
            return characters

        except Exception as e:
            logger.error(f"Error getting active characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_character(self, character_id: str) -> Character:
        """Get a specific character by ID."""
        try:
            response = await self.supabase.table("characters")\
                .select("*")\
                .eq("id", character_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Character not found")

            row = response.data[0]
            character_data = {
                "id": row["id"],
                "name": row["name"],
                "voice_id": row.get("voice_id") or "",
                "global_roleplay": row.get("global_roleplay") or "",
                "system_prompt": row.get("system_prompt") or "",
                "image_url": row.get("image_url") or "",
                "images": row.get("images") or [],
                "is_active": row.get("is_active") or False,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }

            return Character(**character_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_character(self, character_data: CharacterCreate) -> Character:
        """Create a new character."""
        try:
            character_id = await self._generate_character_id(character_data.name)

            db_data = {
                "id": character_id,
                "name": character_data.name,
                "voice_id": character_data.voice_id,
                "global_roleplay": character_data.global_roleplay,
                "system_prompt": character_data.system_prompt,
                "image_url": character_data.image_url,
                "images": character_data.images,
                "is_active": character_data.is_active
            }

            response = await self.supabase.table("characters")\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create character")

            return await self.get_character(character_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_character(self, character_id: str, character_data: CharacterUpdate) -> Character:
        """Update an existing character."""
        try:
            update_data = {}
            if character_data.name is not None:
                update_data["name"] = character_data.name
            if character_data.voice_id is not None:
                update_data["voice_id"] = character_data.voice_id
            if character_data.global_roleplay is not None:
                update_data["global_roleplay"] = character_data.global_roleplay
            if character_data.system_prompt is not None:
                update_data["system_prompt"] = character_data.system_prompt
            if character_data.image_url is not None:
                update_data["image_url"] = character_data.image_url
            if character_data.images is not None:
                update_data["images"] = character_data.images
            if character_data.is_active is not None:
                update_data["is_active"] = character_data.is_active

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = await self.supabase.table("characters")\
                .update(update_data)\
                .eq("id", character_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Character not found")

            return await self.get_character(character_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def set_character_active(self, character_id: str, is_active: bool) -> Character:
        """Set character active status."""
        return await self.update_character(character_id, CharacterUpdate(is_active=is_active))

    async def delete_character(self, character_id: str) -> bool:
        """Delete a character."""
        try:
            await self.get_character(character_id)

            await self.supabase.table("characters")\
                .delete()\
                .eq("id", character_id)\
                .execute()

            logger.info(f"Deleted character: {character_id}")
            return True

        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def search_characters(self, query: str) -> List[Character]:
        """Search characters by name."""
        try:
            response = await self.supabase.table("characters")\
                .select("*")\
                .ilike("name", f"%{query}%")\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice_id": row.get("voice_id") or "",
                    "global_roleplay": row.get("global_roleplay") or "",
                    "system_prompt": row.get("system_prompt") or "",
                    "image_url": row.get("image_url") or "",
                    "images": row.get("images") or [],
                    "is_active": row.get("is_active") or False,
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
                characters.append(Character(**character_data))

            logger.info(f"Found {len(characters)} characters matching '{query}'")
            return characters

        except Exception as e:
            logger.error(f"Error searching characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    ########################################
    ##--        Voice Operations        --##
    ########################################

    async def generate_voice_id(self, voice_name: str) -> str:
        """Generate a sequential ID from the voice display name."""
        base_id = voice_name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)
        base_id = re.sub(r'\s+', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        base_id = base_id.strip('-')
        if not base_id:
            base_id = "voice"

        try:
            response = await self.supabase.table("voices")\
                .select("voice_id")\
                .like("voice_id", f"{base_id}-%")\
                .execute()

            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")

            for row in response.data or []:
                match = pattern.match(row["voice_id"])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)

            next_num = highest_num + 1
            voice_id = f"{base_id}-{next_num:03d}"

            logger.info(f"Generated voice id: {voice_id}")
            return voice_id

        except Exception as e:
            logger.error(f"Error generating voice id: {e}")
            return f"{base_id}-001"


    def _parse_voice_row(self, row: Dict[str, Any]) -> Voice:
        """Helper to parse a database row into a Voice model."""
        return Voice(
            voice_id=row["voice_id"],
            voice_name=row.get("voice_name") or "",
            method=row.get("method") or "",
            ref_audio=row.get("ref_audio") or "",
            ref_text=row.get("ref_text") or "",
            speaker_desc=row.get("speaker_desc") or "",
            scene_prompt=row.get("scene_prompt") or "",
            audio_ids=row.get("audio_ids"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at")
        )

    async def get_all_voices(self) -> List[Voice]:
        """Get all voices from database."""
        try:
            response = await self.supabase.table("voices")\
                .select("*")\
                .execute()

            rows = response.data or []
            voices = [self._parse_voice_row(row) for row in rows]
            voices.sort(key=lambda voice: (voice.voice_name or "").lower())

            logger.info(f"Retrieved {len(voices)} voices from database")
            return voices

        except Exception as e:
            logger.error(f"Error getting all voices: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_voice(self, voice_id: str) -> Voice:
        """Get a specific voice by voice_id (PK)."""
        # Check cache first
        with self._cache_lock:
            if voice_id in self._voice_cache:
                logger.debug(f"Retrieved voice {voice_id} from cache")
                return self._voice_cache[voice_id]["config"]

        try:
            response = await self.supabase.table("voices")\
                .select("*")\
                .eq("voice_id", voice_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")

            row = response.data[0]
            voice = self._parse_voice_row(row)

            # Add to cache
            with self._cache_lock:
                self._voice_cache[voice_id] = {
                    "config": voice,
                    "audio_ids": voice.audio_ids
                }

            logger.info(f"Retrieved voice {voice_id} from database")
            return voice

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_voice(self, voice_data: VoiceCreate) -> Voice:
        """Create a new voice."""
        try:
            db_data = {
                "voice_id": voice_data.voice_id,
                "voice_name": voice_data.voice_name,
                "method": voice_data.method,
                "ref_audio": voice_data.ref_audio,
                "ref_text": voice_data.ref_text,
                "speaker_desc": voice_data.speaker_desc,
                "scene_prompt": voice_data.scene_prompt
            }

            response = await self.supabase.table("voices")\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create voice")

            voice = await self.get_voice(voice_data.voice_id)

            # Add to cache
            with self._cache_lock:
                self._voice_cache[voice_data.voice_id] = {
                    "config": voice,
                    "audio_ids": voice.audio_ids
                }

            logger.info(f"Created voice: {voice.voice_id}")
            return voice

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating voice: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_voice(self, voice_id: str, voice_data: VoiceUpdate) -> Voice:
        """Update an existing voice."""
        try:
            update_data = {}
            if voice_data.voice_name is not None:
                update_data["voice_name"] = voice_data.voice_name
            if voice_data.method is not None:
                update_data["method"] = voice_data.method
            if voice_data.ref_audio is not None:
                update_data["ref_audio"] = voice_data.ref_audio
            if voice_data.ref_text is not None:
                update_data["ref_text"] = voice_data.ref_text
            if voice_data.speaker_desc is not None:
                update_data["speaker_desc"] = voice_data.speaker_desc
            if voice_data.scene_prompt is not None:
                update_data["scene_prompt"] = voice_data.scene_prompt
            if voice_data.audio_ids is not None:
                update_data["audio_ids"] = voice_data.audio_ids

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = await self.supabase.table("voices")\
                .update(update_data)\
                .eq("voice_id", voice_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")

            voice = await self.get_voice(voice_id)

            # Update cache
            with self._cache_lock:
                if voice_id in self._voice_cache:
                    self._voice_cache[voice_id]["config"] = voice
                    if voice_data.audio_ids is not None:
                        self._voice_cache[voice_id]["audio_ids"] = voice_data.audio_ids

            logger.info(f"Updated voice: {voice_id}")
            return voice

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice."""
        try:
            await self.get_voice(voice_id)

            await self.supabase.table("voices")\
                .delete()\
                .eq("voice_id", voice_id)\
                .execute()

            # Remove from cache
            with self._cache_lock:
                if voice_id in self._voice_cache:
                    del self._voice_cache[voice_id]

            logger.info(f"Deleted voice: {voice_id}")
            return True

        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def get_cached_audio_ids(self, voice_id: str) -> Optional[Any]:
        """Get audio_ids from cache if available."""
        with self._cache_lock:
            if voice_id in self._voice_cache:
                return self._voice_cache[voice_id]["audio_ids"]
        return None

    def update_cached_audio_ids(self, voice_id: str, audio_ids: Any):
        """Update audio_ids in cache."""
        with self._cache_lock:
            if voice_id in self._voice_cache:
                self._voice_cache[voice_id]["audio_ids"] = audio_ids

    def clear_voice_cache(self):
        """Clear the voice cache."""
        with self._cache_lock:
            self._voice_cache.clear()
        logger.info("Voice cache cleared")

    ########################################
    ##--    Conversation Operations     --##
    ########################################

    def _generate_conversation_title(self, first_message: Optional[str] = None) -> str:
        """Generate a conversation title."""
        if first_message and first_message.strip():
            title = first_message.strip()[:50]
            if len(first_message.strip()) > 50:
                title += "..."
            return title
        return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async def create_conversation(
        self,
        conversation_data: ConversationCreate,
        auto_generate_title: bool = True
    ) -> Conversation:
        """Create a new conversation."""
        try:
            title = conversation_data.title
            if auto_generate_title and not title:
                title = self._generate_conversation_title()

            db_data = {
                "title": title,
                "active_characters": conversation_data.active_characters or []
            }

            response = await self.supabase.table("conversations")\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create conversation")

            row = response.data[0]
            conversation = Conversation(
                conversation_id=str(row["conversation_id"]),
                title=row.get("title"),
                active_characters=row.get("active_characters") or [],
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Created conversation {conversation.conversation_id}")
            return conversation

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a specific conversation by ID."""
        try:
            response = await self.supabase.table("conversations")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Conversation not found")

            row = response.data[0]
            conversation = Conversation(
                conversation_id=str(row["conversation_id"]),
                title=row.get("title"),
                active_characters=row.get("active_characters") or [],
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Retrieved conversation {conversation_id}")
            return conversation

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_all_conversations(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations ordered by most recent first."""
        try:
            query = self.supabase.table("conversations")\
                .select("*")\
                .order("updated_at", desc=True)

            if limit is not None:
                query = query.limit(limit)

            if offset > 0:
                query = query.range(offset, offset + (limit or 1000) - 1)

            response = await query.execute()

            conversations = []
            for row in response.data:
                conversation = Conversation(
                    conversation_id=str(row["conversation_id"]),
                    title=row.get("title"),
                    active_characters=row.get("active_characters") or [],
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                conversations.append(conversation)

            logger.info(f"Retrieved {len(conversations)} conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error getting all conversations: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_conversation(
        self,
        conversation_id: str,
        conversation_data: ConversationUpdate
    ) -> Conversation:
        """Update an existing conversation."""
        try:
            update_data = {}
            if conversation_data.title is not None:
                update_data["title"] = conversation_data.title
            if conversation_data.active_characters is not None:
                update_data["active_characters"] = conversation_data.active_characters

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = await self.supabase.table("conversations")\
                .update(update_data)\
                .eq("conversation_id", conversation_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Conversation not found")

            logger.info(f"Updated conversation {conversation_id}")
            return await self.get_conversation(conversation_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_conversation_title(self, conversation_id: str, title: str) -> Conversation:
        """Update just the title of a conversation."""
        return await self.update_conversation(
            conversation_id,
            ConversationUpdate(title=title)
        )

    async def update_conversation_active_characters(
        self,
        conversation_id: str,
        active_characters: List[Dict[str, Any]]
    ) -> Conversation:
        """Update the active characters in a conversation."""
        return await self.update_conversation(
            conversation_id,
            ConversationUpdate(active_characters=active_characters)
        )

    async def add_character_to_conversation(
        self,
        conversation_id: str,
        character_data: Dict[str, Any]
    ) -> Conversation:
        """Add a character to the conversation's active_characters list."""
        try:
            conversation = await self.get_conversation(conversation_id)

            # Check if character already exists by ID
            character_ids = [c.get("id") for c in conversation.active_characters]
            if character_data.get("id") not in character_ids:
                active_characters = conversation.active_characters + [character_data]
                return await self.update_conversation_active_characters(conversation_id, active_characters)

            return conversation

        except Exception as e:
            logger.error(f"Error adding character to conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def remove_character_from_conversation(
        self,
        conversation_id: str,
        character_id: str
    ) -> Conversation:
        """Remove a character from the conversation's active_characters list."""
        try:
            conversation = await self.get_conversation(conversation_id)

            active_characters = [c for c in conversation.active_characters if c.get("id") != character_id]
            return await self.update_conversation_active_characters(conversation_id, active_characters)

        except Exception as e:
            logger.error(f"Error removing character from conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation (messages will be cascade deleted)."""
        try:
            await self.get_conversation(conversation_id)

            await self.supabase.table("conversations")\
                .delete()\
                .eq("conversation_id", conversation_id)\
                .execute()

            logger.info(f"Deleted conversation {conversation_id}")
            return True

        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def auto_update_conversation_title(
        self,
        conversation_id: str,
        first_message: str
    ) -> Conversation:
        """Auto-generate and update title from the first message."""
        try:
            conversation = await self.get_conversation(conversation_id)

            if not conversation.title or "Conversation" in conversation.title:
                new_title = self._generate_conversation_title(first_message)
                return await self.update_conversation_title(conversation_id, new_title)

            return conversation

        except Exception as e:
            logger.error(f"Error auto-updating title for conversation {conversation_id}: {e}")
            return conversation

    ########################################
    ##--       Message Operations       --##
    ########################################

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Create a single message."""
        try:
            db_data = {
                "conversation_id": message_data.conversation_id,
                "role": message_data.role,
                "content": message_data.content,
                "name": message_data.name,
                "character_id": message_data.character_id
            }

            response = await self.supabase.table("messages")\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create message")

            row = response.data[0]
            message = Message(
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                role=row["role"],
                name=row.get("name"),
                content=row["content"],
                character_id=row.get("character_id"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Created message {message.message_id} in conversation {message.conversation_id}")
            return message

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_messages_batch(self, messages: List[MessageCreate]) -> List[Message]:
        """Create multiple messages in a single batch operation."""
        try:
            db_data = [
                {
                    "conversation_id": msg.conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "character_id": msg.character_id
                }
                for msg in messages
            ]

            response = await self.supabase.table("messages")\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create messages")

            created_messages = []
            for row in response.data:
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                created_messages.append(message)

            logger.info(f"Created {len(created_messages)} messages in batch")
            return created_messages

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating messages batch: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation with optional pagination."""
        try:
            query = self.supabase.table("messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=False)

            if limit is not None:
                query = query.limit(limit)

            if offset > 0:
                query = query.range(offset, offset + (limit or 1000) - 1)

            response = await query.execute()

            messages = []
            for row in response.data:
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                messages.append(message)

            logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_recent_messages(self, conversation_id: str, n: int = 10) -> List[Message]:
        """Get the last N messages from a conversation."""
        try:
            response = await self.supabase.table("messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=True)\
                .limit(n)\
                .execute()

            # Reverse to get chronological order
            messages = []
            for row in reversed(response.data):
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                messages.append(message)

            logger.info(f"Retrieved last {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting recent messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_last_message(self, conversation_id: str) -> Optional[Message]:
        """Get the last message from a conversation."""
        try:
            response = await self.supabase.table("messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()

            if not response.data:
                return None

            row = response.data[0]
            message = Message(
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                role=row["role"],
                name=row.get("name"),
                content=row["content"],
                character_id=row.get("character_id"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            return message

        except Exception as e:
            logger.error(f"Error getting last message for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_message_count(self, conversation_id: str) -> int:
        """Get the total number of messages in a conversation."""
        try:
            response = await self.supabase.table("messages")\
                .select("message_id", count="exact")\
                .eq("conversation_id", conversation_id)\
                .execute()

            count = response.count if hasattr(response, 'count') and response.count is not None else len(response.data)
            logger.info(f"Conversation {conversation_id} has {count} messages")
            return count

        except Exception as e:
            logger.error(f"Error getting message count for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_message(self, message_id: str) -> bool:
        """Delete a single message."""
        try:
            response = await self.supabase.table("messages")\
                .delete()\
                .eq("message_id", message_id)\
                .execute()

            logger.info(f"Deleted message {message_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_messages_for_conversation(self, conversation_id: str) -> bool:
        """Delete all messages for a conversation."""
        try:
            await self.supabase.table("messages")\
                .delete()\
                .eq("conversation_id", conversation_id)\
                .execute()

            logger.info(f"Deleted messages for conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


########################################
##--      Module-Level Instance     --##
########################################

# Create a default instance for easy importing
db = Database()