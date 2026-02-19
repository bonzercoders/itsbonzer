import { supabase } from '@/lib/supabase'
import type { Character } from '@/components/characters/types'

/**
 * Map a Supabase row (snake_case) → frontend Character (camelCase).
 */
function fromDb(row: Record<string, unknown>): Character {
  return {
    id: row.id as string,
    name: row.name as string,
    systemPrompt: (row.system_prompt as string) ?? '',
    globalPrompt: (row.global_roleplay as string) ?? '',
    voiceId: (row.voice_id as string) ?? undefined,
    imageDataUrl: (row.image_url as string) ?? undefined,
    isActive: (row.is_active as boolean) ?? false,
  }
}

/**
 * Map a frontend Character → Supabase row for insert/update.
 */
function toDb(c: Character) {
  return {
    id: c.id,
    name: c.name,
    system_prompt: c.systemPrompt,
    global_roleplay: c.globalPrompt,
    voice_id: c.voiceId ?? null,
    image_url: c.imageDataUrl ?? null,
    is_active: c.isActive ?? false,
  }
}

/**
 * Generate a slug-based character ID (e.g. "aria-001").
 * Mirrors the backend's _generate_character_id algorithm.
 */
export async function generateCharacterId(name: string): Promise<string> {
  let baseId = name.toLowerCase().trim()
  baseId = baseId.replace(/[^a-z0-9\s-]/g, '')
  baseId = baseId.replace(/\s+/g, '-')
  baseId = baseId.replace(/-+/g, '-')
  baseId = baseId.replace(/^-|-$/g, '')

  if (!baseId) baseId = 'character'

  const { data } = await supabase
    .from('characters')
    .select('id')
    .like('id', `${baseId}-%`)

  let highestNum = 0
  const pattern = new RegExp(`^${baseId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-(\\d{3})$`)

  for (const row of data ?? []) {
    const match = pattern.exec(row.id)
    if (match) {
      highestNum = Math.max(highestNum, parseInt(match[1], 10))
    }
  }

  return `${baseId}-${String(highestNum + 1).padStart(3, '0')}`
}

export async function fetchCharacters(): Promise<Character[]> {
  const { data, error } = await supabase
    .from('characters')
    .select('*')
    .order('created_at', { ascending: false })

  if (error) throw error
  return (data ?? []).map(fromDb)
}

export async function createCharacter(c: Character): Promise<Character> {
  const { data, error } = await supabase
    .from('characters')
    .insert(toDb(c))
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function updateCharacter(c: Character): Promise<Character> {
  const { data, error } = await supabase
    .from('characters')
    .update(toDb(c))
    .eq('id', c.id)
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}

export async function deleteCharacter(id: string): Promise<void> {
  const { error } = await supabase
    .from('characters')
    .delete()
    .eq('id', id)

  if (error) throw error
}

export async function setCharacterActive(id: string, isActive: boolean): Promise<Character> {
  const { data, error } = await supabase
    .from('characters')
    .update({ is_active: isActive })
    .eq('id', id)
    .select()
    .single()

  if (error) throw error
  return fromDb(data)
}
