import { supabase } from '@/lib/supabase'
import type { Voice } from '@/components/speech/types'

/**
 * Map a Supabase row (DB columns) → frontend Voice type.
 */
function fromDb(row: Record<string, unknown>): Voice {
  return {
    voiceId: row.voice_id as string,        // DB PK
    voiceName: (row.voice_name as string) ?? '',
    method: (row.method as Voice['method']) === 'profile' ? 'profile' : 'clone',
    scenePrompt: (row.scene_prompt as string) ?? '',
    referenceText: (row.ref_text as string) ?? '',
    referenceAudio: (row.ref_audio as string) ?? '',
    speakerDescription: (row.speaker_desc as string) ?? '',
  }
}

/**
 * Map a frontend Voice → Supabase row for insert/update.
 */
function toDb(v: Voice) {
  return {
    voice_id: v.voiceId,      // PK
    voice_name: v.voiceName,  // Display name
    method: v.method,
    scene_prompt: v.scenePrompt,
    ref_text: v.referenceText,
    ref_audio: v.referenceAudio,
    speaker_desc: v.speakerDescription,
  }
}

/**
 * Generate a slug-based voice ID (e.g. "amelia-001").
 * Mirrors the backend generate_voice_id algorithm.
 */
export async function generateVoiceId(voiceName: string): Promise<string> {
  let baseId = voiceName.toLowerCase().trim()
  baseId = baseId.replace(/[^a-z0-9\s-]/g, '')
  baseId = baseId.replace(/\s+/g, '-')
  baseId = baseId.replace(/-+/g, '-')
  baseId = baseId.replace(/^-|-$/g, '')

  if (!baseId) baseId = 'voice'

  const { data } = await supabase
    .from('voices')
    .select('voice_id')
    .like('voice_id', `${baseId}-%`)

  let highestNum = 0
  const pattern = new RegExp(`^${baseId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-(\\d{3})$`)

  for (const row of data ?? []) {
    const voiceId = row.voice_id as string
    const match = pattern.exec(voiceId)
    if (match) {
      highestNum = Math.max(highestNum, parseInt(match[1], 10))
    }
  }

  return `${baseId}-${String(highestNum + 1).padStart(3, '0')}`
}

export async function fetchVoices(): Promise<Voice[]> {
  const { data, error } = await supabase
    .from('voices')
    .select('*')
    .order('created_at', { ascending: false })

  if (error) throw error
  return (data ?? []).map(fromDb)
}

export async function createVoice(v: Voice): Promise<Voice> {
  const { data, error } = await supabase
    .from('voices')
    .insert(toDb(v))
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function updateVoice(v: Voice): Promise<Voice> {
  const { data, error } = await supabase
    .from('voices')
    .update(toDb(v))
    .eq('voice_id', v.voiceId)
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function deleteVoice(voiceId: string): Promise<void> {
  const { error } = await supabase
    .from('voices')
    .delete()
    .eq('voice_id', voiceId)

  if (error) throw error
}
