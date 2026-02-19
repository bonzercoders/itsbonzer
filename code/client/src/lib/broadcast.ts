import { supabase } from '@/lib/supabase'
import type { RealtimeChannel } from '@supabase/supabase-js'

type BroadcastChannelName = 'db-characters' | 'db-voices'

const channelPromises: Partial<Record<BroadcastChannelName, Promise<RealtimeChannel>>> = {}

function invalidateChannel(name: BroadcastChannelName, channel: RealtimeChannel) {
  delete channelPromises[name]
  void supabase.removeChannel(channel)
}

async function getBroadcastChannel(name: BroadcastChannelName): Promise<RealtimeChannel> {
  const existing = channelPromises[name]
  if (existing) {
    return existing
  }

  const channelPromise = new Promise<RealtimeChannel>((resolve, reject) => {
    const channel = supabase.channel(name, {
      config: { broadcast: { ack: true, self: false } },
    })

    let settled = false

    channel.subscribe((status, err) => {
      if (status === 'SUBSCRIBED') {
        settled = true
        resolve(channel)
        return
      }

      if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT' || status === 'CLOSED') {
        if (!settled) {
          settled = true
          invalidateChannel(name, channel)
          reject(err ?? new Error(`Failed to subscribe to ${name}`))
        }
      }
    })
  })

  channelPromises[name] = channelPromise
  return channelPromise
}

/**
 * Notify the backend that a character was created, updated, or deleted.
 * The backend subscribes to 'db-characters' and refreshes ChatLLM.active_characters.
 */
export async function broadcastCharacterChange(
  action: 'created' | 'updated' | 'deleted',
  characterId: string
) {
  const channel = await getBroadcastChannel('db-characters')
  const status = await channel.send({
    type: 'broadcast',
    event: 'character-changed',
    payload: { action, characterId },
  })

  if (status !== 'ok') {
    invalidateChannel('db-characters', channel)
    throw new Error(`Character broadcast failed with status: ${status}`)
  }
}

/**
 * Notify the backend that a voice was created, updated, or deleted.
 * The backend subscribes to 'db-voices' and clears its voice cache.
 */
export async function broadcastVoiceChange(
  action: 'created' | 'updated' | 'deleted',
  voiceId: string
) {
  const channel = await getBroadcastChannel('db-voices')
  const status = await channel.send({
    type: 'broadcast',
    event: 'voice-changed',
    payload: { action, voiceId },
  })

  if (status !== 'ok') {
    invalidateChannel('db-voices', channel)
    throw new Error(`Voice broadcast failed with status: ${status}`)
  }
}
