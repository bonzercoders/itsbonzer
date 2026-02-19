export type Character = {
  id: string
  name: string
  systemPrompt: string
  globalPrompt: string
  voiceId?: string
  imageDataUrl?: string
  isActive?: boolean
}
