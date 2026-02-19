import { Plus, Search, Volume2 } from 'lucide-react'
import { useMemo, useState, type KeyboardEvent } from 'react'

import { type Voice } from '@/components/speech/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemSeparator,
  ItemTitle,
} from '@/components/ui/item'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'

type VoiceDirectoryProps = {
  voices: Voice[]
  selectedId?: string | null
  onSelect: (voiceId: string) => void
  onCreate: () => void
}

const alphabet = [
  '#',
  ...Array.from({ length: 26 }, (_, index) =>
    String.fromCharCode(65 + index)
  ),
]

const groupVoicesByLetter = (items: Voice[]) =>
  items.reduce<Record<string, Voice[]>>((groups, voiceRecord) => {
    const letter = voiceRecord.voiceName.trim().charAt(0).toUpperCase() || '#'
    if (!groups[letter]) {
      groups[letter] = []
    }
    groups[letter].push(voiceRecord)
    return groups
  }, {})

const getLetterAnchor = (letter: string) => (letter === '#' ? 'misc' : letter)

const getDescriptionPreview = (description: string, limit = 100) => {
  const trimmed = description.trim()
  if (!trimmed) {
    return ''
  }
  return trimmed.slice(0, limit)
}

const methodBadgeStyles: Record<Voice['method'], string> = {
  clone: 'border-red-500/30 bg-red-500/10 text-red-200',
  profile: 'border-[#a855f7]/30 bg-[#a855f7]/10 text-[#e9d5ff]',
}

function VoiceDirectory({
  voices,
  selectedId,
  onSelect,
  onCreate,
}: VoiceDirectoryProps) {
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()

  const filteredVoices = useMemo(() => {
    if (!normalizedQuery) {
      return voices
    }

    return voices.filter((voiceRecord) => {
      const haystack = `${voiceRecord.voiceName} ${voiceRecord.method} ${voiceRecord.speakerDescription}`.toLowerCase()
      return haystack.includes(normalizedQuery)
    })
  }, [normalizedQuery, voices])

  const groupedVoices = useMemo(() => {
    const grouped = groupVoicesByLetter(filteredVoices)
    Object.keys(grouped).forEach((letter) => {
      grouped[letter] = grouped[letter].slice().sort((a, b) => {
        return a.voiceName.localeCompare(b.voiceName)
      })
    })
    return grouped
  }, [filteredVoices])

  const lettersWithVoices = useMemo(
    () => alphabet.filter((letter) => groupedVoices[letter]?.length),
    [groupedVoices]
  )

  const jumpToLetter = (letter: string) => {
    const target = document.getElementById(
      `voice-letter-${getLetterAnchor(letter)}`
    )
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const handleItemKeyDown = (
    event: KeyboardEvent<HTMLDivElement>,
    voiceId: string
  ) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(voiceId)
    }
  }

  return (
    <div className="flex h-full w-full min-h-0 flex-col">
      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[#2d3138] bg-[#171a1f]/85',
          'shadow-[0_20px_40px_rgba(0,0,0,0.35)]'
        )}
      >
        <div className="flex flex-wrap items-center gap-3 border-b border-[#2a2f36] px-4 py-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#7d8792]" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search voices..."
              aria-label="Search voices"
              className={cn(
                'h-9 border-[#2d3138] bg-[#171a1f] pl-9 text-sm text-[#dfe3e8]',
                'placeholder:text-[#6c7480] focus-visible:border-[#3a414b] focus-visible:ring-[#2c323a]'
              )}
            />
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            aria-label="Create voice"
            className={cn(
              'border-[#2f353d] bg-[#1e2228] text-[#cbd2da]',
              'hover:border-[#3b424c] hover:bg-[#252a32] hover:text-white'
            )}
            onClick={onCreate}
          >
            <Plus className="h-4 w-4 text-[#7fd2ff]" />
          </Button>
        </div>

        <div className="flex min-h-0 flex-1">
          <ScrollArea className="flex-1">
            <div className="px-5 py-4">
              {lettersWithVoices.length ? (
                <div className="space-y-8">
                  {lettersWithVoices.map((letter) => {
                    const items = groupedVoices[letter] ?? []
                    return (
                      <section
                        key={letter}
                        id={`voice-letter-${getLetterAnchor(letter)}`}
                        className="scroll-mt-6 space-y-3"
                      >
                        <div className="text-[11px] font-semibold uppercase tracking-[0.4em] text-[#8b93a0]">
                          {letter}
                        </div>
                        <ItemGroup>
                          {items.map((voiceRecord, index) => {
                            const isSelected = selectedId === voiceRecord.voiceId
                            const descriptionPreview = getDescriptionPreview(
                              voiceRecord.speakerDescription
                            )
                            return (
                              <div key={voiceRecord.voiceId}>
                                <Item
                                  size="sm"
                                  role="button"
                                  tabIndex={0}
                                  aria-pressed={isSelected}
                                  onClick={() => onSelect(voiceRecord.voiceId)}
                                  onKeyDown={(event) =>
                                    handleItemKeyDown(event, voiceRecord.voiceId)
                                  }
                                  className={cn(
                                    'border-[#2a2f36] bg-[#1a1d22]/80',
                                    'cursor-pointer hover:bg-[#22262d]',
                                    'focus-visible:border-[#4aa3ff] focus-visible:ring-[#4aa3ff]/40',
                                    isSelected
                                      ? 'border-[#4aa3ff]/60 bg-[#1d222b] shadow-[0_0_0_1px_rgba(74,163,255,0.25)]'
                                      : null
                                  )}
                                >
                                  <ItemContent>
                                    <ItemTitle className="w-full">
                                      <span className="text-sm font-semibold text-[#e2e6ea]">
                                        {voiceRecord.voiceName}
                                      </span>
                                      <div className="ml-auto flex flex-wrap items-center gap-2">
                                        <Badge
                                          variant="outline"
                                          className={cn(
                                            'border text-[10px] uppercase tracking-[0.2em]',
                                            methodBadgeStyles[voiceRecord.method]
                                          )}
                                        >
                                          {voiceRecord.method}
                                        </Badge>
                                      </div>
                                    </ItemTitle>
                                    {descriptionPreview ? (
                                      <ItemDescription className="text-[#8b93a0]">
                                        {descriptionPreview}
                                      </ItemDescription>
                                    ) : null}
                                  </ItemContent>
                                  <ItemActions>
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      className={cn(
                                        'border-[#2f353d] bg-[#1e2228] text-[#cbd2da]',
                                        'hover:border-[#3b424c] hover:bg-[#252a32] hover:text-white'
                                      )}
                                    >
                                      <Volume2 className="h-4 w-4 text-[#7fd2ff]" />
                                      Preview
                                    </Button>
                                  </ItemActions>
                                </Item>
                                {index < items.length - 1 ? (
                                  <ItemSeparator className="my-2 bg-[#2a2f36]" />
                                ) : null}
                              </div>
                            )}
                          )},
                        </ItemGroup>
                      </section>
                    )
                  })}
                </div>
              ) : (
                <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-2 text-center text-sm text-[#7a828c]">
                  {voices.length ? (
                    <span>No voices match your search.</span>
                  ) : (
                    <>
                      <span className="text-[#a5adb8]">No voices yet.</span>
                      <span>Create one to start building.</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </ScrollArea>
          <div className="hidden w-10 flex-col items-center gap-1 border-l border-[#242a31] py-4 pr-3 text-[10px] font-semibold text-[#6f7782] md:flex">
            {alphabet.map((letter) => {
              const isActive = groupedVoices[letter]?.length
              return (
                <button
                  key={letter}
                  type="button"
                  onClick={() => jumpToLetter(letter)}
                  disabled={!isActive}
                  aria-label={`Jump to ${letter}`}
                  className={cn(
                    'rounded-full px-1 py-0.5 transition',
                    isActive
                      ? 'text-[#8b93a0] hover:bg-[#242a31] hover:text-white'
                      : 'text-[#404751] opacity-50'
                  )}
                >
                  {letter}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default VoiceDirectory
