import { useCallback, useEffect, useState } from 'react'

import VoiceBuilderForm from '@/components/speech/VoiceBuilderForm'
import VoiceDirectory from '@/components/speech/VoiceDirectory'
import { type Voice } from '@/components/speech/types'
import {
  fetchVoices,
  generateVoiceId,
  createVoice as apiCreateVoice,
  updateVoice as apiUpdateVoice,
  deleteVoice as apiDeleteVoice,
} from '@/lib/api/voices'
import { broadcastVoiceChange } from '@/lib/broadcast'

type DraftState = {
  draft: Voice
  isNew: boolean
}

const createVoiceDraft = (voiceId: string): Voice => ({
  voiceId,
  voiceName: '',
  method: 'clone',
  scenePrompt: '',
  referenceText: '',
  referenceAudio: '',
  speakerDescription: '',
})

function SpeechPage() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeDraft, setActiveDraft] = useState<DraftState | null>(null)

  // Load voices from Supabase on mount
  useEffect(() => {
    fetchVoices()
      .then(setVoices)
      .catch((err) => console.error('Failed to load voices:', err))
      .finally(() => setLoading(false))
  }, [])

  const handleCreate = () => {
    const voiceId = `temp-${Date.now()}`
    const newVoice = createVoiceDraft(voiceId)
    setActiveDraft({ draft: newVoice, isNew: true })
  }

  const handleSelect = (id: string) => {
    const selected = voices.find((voice) => voice.voiceId === id)
    if (!selected) {
      return
    }
    setSelectedId(id)
    setActiveDraft({ draft: { ...selected }, isNew: false })
  }

  const handleDraftChange = (updates: Partial<Voice>) => {
    setActiveDraft((previous) => {
      if (!previous) {
        return previous
      }
      return { ...previous, draft: { ...previous.draft, ...updates } }
    })
  }

  const handleDelete = useCallback(async () => {
    if (!activeDraft) {
      return
    }
    if (activeDraft.isNew) {
      setActiveDraft(null)
      return
    }

    const { voiceId } = activeDraft.draft
    try {
      await apiDeleteVoice(voiceId)
      setVoices((previous) => previous.filter((v) => v.voiceId !== voiceId))
      try {
        await broadcastVoiceChange('deleted', voiceId)
      } catch (broadcastError) {
        console.warn('Voice delete broadcast failed:', broadcastError)
      }
    } catch (err) {
      console.error('Failed to delete voice:', err)
    }
    if (selectedId === voiceId) {
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
        const voiceId = await generateVoiceId(draft.voiceName || 'voice')
        const voiceToSave = { ...draft, voiceId }
        const saved = await apiCreateVoice(voiceToSave)
        setVoices((previous) => [saved, ...previous])
        try {
          await broadcastVoiceChange('created', saved.voiceId)
        } catch (broadcastError) {
          console.warn('Voice create broadcast failed:', broadcastError)
        }
      } else {
        const saved = await apiUpdateVoice(draft)
        setVoices((previous) =>
          previous.map((voice) => (voice.voiceId === saved.voiceId ? saved : voice))
        )
        try {
          await broadcastVoiceChange('updated', saved.voiceId)
        } catch (broadcastError) {
          console.warn('Voice update broadcast failed:', broadcastError)
        }
      }
    } catch (err) {
      console.error('Failed to save voice:', err)
    }
    setSelectedId(null)
    setActiveDraft(null)
  }, [activeDraft])

  if (loading) {
    return (
      <div className="flex h-full w-full items-center justify-center text-[#7a828c]">
        Loading voices...
      </div>
    )
  }

  return (
    <div className="h-full w-full p-6">
      <div className="grid h-full min-h-0 w-full gap-8 lg:grid-cols-[1fr_3fr]">
        <div className="flex h-full min-h-0 w-full flex-col panel-fade-in">
        <VoiceDirectory
          voices={voices}
          selectedId={selectedId}
          onSelect={handleSelect}
          onCreate={handleCreate}
        />
      </div>
      <div className="flex w-full flex-1 flex-col">
        {activeDraft ? (
          <div className="h-full w-full animate-in fade-in duration-200">
            <VoiceBuilderForm
              key={activeDraft.draft.voiceId}
              voiceDraft={activeDraft.draft}
              onChange={handleDraftChange}
              onDelete={handleDelete}
              onSave={handleSave}
            />
          </div>
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-[#2b3139] bg-[#13161a]/40 text-sm text-[#7a828c]">
            Select a voice to edit or create a new one.
          </div>
        )}
      </div>
    </div>
  </div>
  )
}

export default SpeechPage
