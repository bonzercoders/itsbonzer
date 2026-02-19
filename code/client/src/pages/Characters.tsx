import { useCallback, useEffect, useState } from 'react'

import CharacterDirectory from '@/components/characters/CharacterDirectory'
import CharacterEditor from '@/components/characters/CharacterEditor'
import { type Character } from '@/components/characters/types'
import {
  fetchCharacters,
  createCharacter as apiCreateCharacter,
  updateCharacter as apiUpdateCharacter,
  deleteCharacter as apiDeleteCharacter,
  generateCharacterId,
  setCharacterActive,
} from '@/lib/api/characters'
import { broadcastCharacterChange } from '@/lib/broadcast'
import { fetchVoices } from '@/lib/api/voices'
import { type Voice } from '@/components/speech/types'

type VoiceOption = {
  value: string
  label: string
}

type DraftState = {
  draft: Character
  isNew: boolean
}

const createCharacter = (id: string): Character => ({
  id,
  name: '',
  systemPrompt: '',
  globalPrompt: '',
  voiceId: undefined,
  imageDataUrl: undefined,
})

function CharactersPage() {
  const [characters, setCharacters] = useState<Character[]>([])
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeDraft, setActiveDraft] = useState<DraftState | null>(null)

  // Load characters from Supabase on mount
  useEffect(() => {
    fetchCharacters()
      .then(setCharacters)
      .catch((err) => console.error('Failed to load characters:', err))
      .finally(() => setLoading(false))
  }, [])

  // Load voices from Supabase on mount
  useEffect(() => {
    fetchVoices()
      .then(setVoices)
      .catch((err) => console.error('Failed to load voices:', err))
  }, [])

  const handleCreate = async () => {
    // Create temporary character with placeholder ID
    const tempId = `temp-${Date.now()}`
    const newCharacter = createCharacter(tempId)
    setActiveDraft({ draft: newCharacter, isNew: true })
  }

  const handleSelect = (id: string) => {
    const selected = characters.find((char) => char.id === id)
    if (!selected) {
      return
    }
    setSelectedId(id)
    setActiveDraft({ draft: { ...selected }, isNew: false })
  }

  const handleDraftChange = (updates: Partial<Character>) => {
    setActiveDraft((previous) => {
      if (!previous) {
        return previous
      }
      return { ...previous, draft: { ...previous.draft, ...updates } }
    })
  }

  const handleClose = useCallback(() => {
    setSelectedId(null)
    setActiveDraft(null)
  }, [])

  const handleDelete = useCallback(async () => {
    if (!activeDraft) {
      return
    }
    if (activeDraft.isNew) {
      setActiveDraft(null)
      return
    }

    const { id } = activeDraft.draft
    try {
      await apiDeleteCharacter(id)
      await broadcastCharacterChange('deleted', id)
      setCharacters((previous) => previous.filter((char) => char.id !== id))
    } catch (err) {
      console.error('Failed to delete character:', err)
    }
    if (selectedId === id) {
      setSelectedId(null)
    }
    setActiveDraft(null)
  }, [activeDraft, selectedId])

  const handleSave = useCallback(async () => {
    if (!activeDraft) {
      return
    }

    const { draft, isNew } = activeDraft
    try {
      if (isNew) {
        // Generate slug-based ID from the character name
        const characterId = await generateCharacterId(draft.name || 'character')
        const characterToSave = { ...draft, id: characterId }

        const saved = await apiCreateCharacter(characterToSave)
        await broadcastCharacterChange('created', saved.id)
        setCharacters((previous) => [saved, ...previous])
      } else {
        const saved = await apiUpdateCharacter(draft)
        await broadcastCharacterChange('updated', saved.id)
        setCharacters((previous) =>
          previous.map((char) => (char.id === saved.id ? saved : char))
        )
      }
    } catch (err) {
      console.error('Failed to save character:', err)
    }
    setSelectedId(null)
    setActiveDraft(null)
  }, [activeDraft])

  const handleChat = useCallback(async (characterId: string) => {
    const character = characters.find((char) => char.id === characterId)
    if (!character) {
      return
    }

    try {
      // Toggle the active status
      const updated = await setCharacterActive(characterId, !character.isActive)
      await broadcastCharacterChange('updated', updated.id)
      setCharacters((previous) =>
        previous.map((char) => (char.id === updated.id ? updated : char))
      )
      // Update the active draft if it's the same character
      if (activeDraft && activeDraft.draft.id === characterId) {
        setActiveDraft({ ...activeDraft, draft: updated })
      }
    } catch (err) {
      console.error('Failed to toggle character active status:', err)
    }
  }, [characters, activeDraft])

  // Transform voices to options for the dropdown
  const voiceOptions: VoiceOption[] = voices.map((voiceRecord) => ({
    value: voiceRecord.voiceId,
    label: voiceRecord.voiceName,
  }))

  if (loading) {
    return (
      <div className="flex h-full w-full items-center justify-center text-[#7a828c]">
        Loading characters...
      </div>
    )
  }

  return (
    <div className="h-full w-full p-6">
      <div className="grid h-full min-h-0 w-full gap-8 lg:grid-cols-[1.25fr_3fr]">
        <div className="flex h-full min-h-0 w-full flex-col panel-fade-in">
        <CharacterDirectory
          characters={characters}
          selectedId={selectedId}
          onSelect={handleSelect}
          onCreate={handleCreate}
          onChat={handleChat}
        />
      </div>
      <div className="flex w-full flex-1 flex-col">
        {activeDraft ? (
          <div className="h-full w-full animate-in fade-in duration-200">
            <CharacterEditor
              key={activeDraft.draft.id}
              character={activeDraft.draft}
              voiceOptions={voiceOptions}
              onChat={() => handleChat(activeDraft.draft.id)}
              onChange={handleDraftChange}
              onClose={handleClose}
              onDelete={handleDelete}
              onSave={handleSave}
            />
          </div>
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-[#2b3139] bg-[#13161a]/40 text-sm text-[#7a828c]">
            Select a character to edit or create a new one.
          </div>
        )}
      </div>
    </div>
    </div>
  )
}

export default CharactersPage
