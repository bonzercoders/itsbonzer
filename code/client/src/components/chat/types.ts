export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  inputMode?: 'typed' | 'stt'
  name: string | null
  characterId: string | null
  content: string
  isStreaming: boolean
  interrupted?: boolean
  realtime?: {
    liveText: string
  }
}
