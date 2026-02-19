import { supabase } from '@/lib/supabase'

export type Conversation = {
  conversationId: string
  title: string | null
  activeCharacters: Record<string, unknown>[]
  createdAt: string | null
  updatedAt: string | null
}

export type Message = {
  messageId: string
  conversationId: string
  role: string
  name: string | null
  content: string
  characterId: string | null
  createdAt: string | null
}

function conversationFromDb(row: Record<string, unknown>): Conversation {
  return {
    conversationId: row.conversation_id as string,
    title: (row.title as string) ?? null,
    activeCharacters: (row.active_characters as Record<string, unknown>[]) ?? [],
    createdAt: (row.created_at as string) ?? null,
    updatedAt: (row.updated_at as string) ?? null,
  }
}

function messageFromDb(row: Record<string, unknown>): Message {
  return {
    messageId: row.message_id as string,
    conversationId: row.conversation_id as string,
    role: row.role as string,
    name: (row.name as string) ?? null,
    content: row.content as string,
    characterId: (row.character_id as string) ?? null,
    createdAt: (row.created_at as string) ?? null,
  }
}

export async function fetchConversations(): Promise<Conversation[]> {
  const { data, error } = await supabase
    .from('conversations')
    .select('*')
    .order('updated_at', { ascending: false })

  if (error) throw error
  return (data ?? []).map(conversationFromDb)
}

export async function fetchMessages(conversationId: string): Promise<Message[]> {
  const { data, error } = await supabase
    .from('messages')
    .select('*')
    .eq('conversation_id', conversationId)
    .order('created_at', { ascending: true })

  if (error) throw error
  return (data ?? []).map(messageFromDb)
}
