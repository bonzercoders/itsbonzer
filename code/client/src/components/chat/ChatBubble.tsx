import type { RefObject } from 'react'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import type { Character } from '@/components/characters/types'
import { cn } from '@/lib/utils'
import type { ChatMessage } from './types'

function getInitials(name: string): string {
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? '')
    .join('')
}

type ChatBubbleProps = {
  message: ChatMessage
  character: Character | null
  streamingRef?: RefObject<HTMLDivElement | null>
}

export function ChatBubble({ message, character, streamingRef }: ChatBubbleProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    const realtime = message.isStreaming ? message.realtime : undefined

    return (
      <div className="flex justify-end px-4 py-1">
        <div
          className={cn(
            'max-w-[75%] rounded-2xl rounded-tr-sm px-4 py-2.5',
            'bg-[#007acc]/20 text-[#dfe3e8]'
          )}
        >
          {realtime ? (
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {realtime.liveText || '\u00a0'}
            </p>
          ) : (
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {message.content}
            </p>
          )}
        </div>
      </div>
    )
  }

  const name = message.name ?? character?.name ?? 'Assistant'
  const initials = getInitials(name)

  return (
    <div className="flex items-start gap-3 px-4 py-1">
      <Avatar className="mt-0.5 h-8 w-8 shrink-0 border border-[#2d3138]">
        {character?.imageDataUrl ? (
          <AvatarImage src={character.imageDataUrl} alt={name} />
        ) : null}
        <AvatarFallback className="bg-[#22262d] text-[10px] font-semibold text-[#8b93a0]">
          {initials}
        </AvatarFallback>
      </Avatar>

      <div className="flex min-w-0 flex-col gap-1">
        <span className="text-xs font-medium text-[#8b93a0]">{name}</span>
        <div
          className={cn(
            'max-w-full rounded-2xl rounded-tl-sm px-4 py-2.5',
            'bg-[#22262d] text-[#dfe3e8]'
          )}
        >
          {message.isStreaming ? (
            <>
              <div
                ref={streamingRef}
                className="whitespace-pre-wrap break-words text-sm leading-relaxed"
              />
              <span className="mt-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-[#7fd2ff] align-middle" />
            </>
          ) : (
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {message.content}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
