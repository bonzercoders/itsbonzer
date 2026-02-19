import { supabase } from '@/lib/supabase'

/**
 * Get a user setting by key. Returns null if not found.
 */
export async function getSetting<T = unknown>(key: string): Promise<T | null> {
  const { data, error } = await supabase
    .from('user_settings')
    .select('setting_value')
    .eq('setting_key', key)
    .maybeSingle()

  if (error) throw error
  return data?.setting_value as T | null
}

/**
 * Upsert a user setting (insert or update by setting_key).
 */
export async function setSetting<T = unknown>(key: string, value: T): Promise<void> {
  // Try update first, then insert if no row matched
  const { data } = await supabase
    .from('user_settings')
    .update({ setting_value: value as unknown })
    .eq('setting_key', key)
    .select('id')

  if (!data || data.length === 0) {
    const { error } = await supabase
      .from('user_settings')
      .insert({ setting_key: key, setting_value: value as unknown })

    if (error) throw error
  }
}
