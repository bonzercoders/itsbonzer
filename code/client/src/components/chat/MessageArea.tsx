import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import type { Character } from '@/components/characters/types'
import { ChatBubble } from './ChatBubble'
import type { ChatMessage } from './types'

type MessageAreaProps = {
  messages: ChatMessage[]
  characterMap: Map<string, Character>
  streamingRef: RefObject<HTMLDivElement | null>
}

export function MessageArea({ messages, characterMap, streamingRef }: MessageAreaProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const isNearBottomRef = useRef(true)

  // Track whether user is near the bottom so we don't hijack scroll during history review
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el
      isNearBottomRef.current = scrollHeight - scrollTop - clientHeight < 100
    }

    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el.removeEventListener('scroll', handleScroll)
  }, [])

  // Auto-scroll when a new message is added
  useEffect(() => {
    if (messages.length === 0) return
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages.length])

  let streamingMessageId: string | null = null
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.isStreaming) {
      streamingMessageId = messages[index].id
      break
    }
  }

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[760px] pb-[300px] pt-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center py-24 text-sm text-[#4b5563]">
          </div>
        ) : (
          messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              character={
                message.characterId ? (characterMap.get(message.characterId) ?? null) : null
              }
              streamingRef={message.id === streamingMessageId ? streamingRef : undefined}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
