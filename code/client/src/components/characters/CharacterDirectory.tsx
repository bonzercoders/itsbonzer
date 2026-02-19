import { MessageCircle, Plus, Search } from 'lucide-react'
import { useMemo, useState, type KeyboardEvent } from 'react'

import { type Character } from '@/components/characters/types'
import { Button } from '@/components/ui/button'
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemSeparator,
  ItemTitle,
} from '@/components/ui/item'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'

type CharacterDirectoryProps = {
  characters: Character[]
  selectedId?: string | null
  onSelect: (characterId: string) => void
  onCreate: () => void
  onChat?: (characterId: string) => void
}

const alphabet = [
  '#',
  ...Array.from({ length: 26 }, (_, index) =>
    String.fromCharCode(65 + index)
  ),
]

const groupCharactersByLetter = (items: Character[]) =>
  items.reduce<Record<string, Character[]>>((groups, character) => {
    const letter = character.name.trim().charAt(0).toUpperCase() || '#'
    if (!groups[letter]) {
      groups[letter] = []
    }
    groups[letter].push(character)
    return groups
  }, {})

const getLetterAnchor = (letter: string) => (letter === '#' ? 'misc' : letter)

const getPromptPreview = (prompt: string, limit = 100) => {
  const trimmed = prompt.trim()
  if (!trimmed) {
    return ''
  }
  if (trimmed.length <= limit) {
    return trimmed
  }
  return `${trimmed.slice(0, limit)}...`
}

const getFallbackInitials = (name: string) => {
  const parts = name.trim().split(/\s+/)
  if (!parts.length) {
    return 'CH'
  }
  const initials = parts
    .slice(0, 2)
    .map((part) => part.charAt(0).toUpperCase())
    .join('')
  return initials || 'CH'
}

function CharacterDirectory({
  characters,
  selectedId,
  onSelect,
  onCreate,
  onChat,
}: CharacterDirectoryProps) {
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()

  const filteredCharacters = useMemo(() => {
    if (!normalizedQuery) {
      return characters
    }

    return characters.filter((character) => {
      const haystack = `${character.name} ${character.systemPrompt}`.toLowerCase()
      return haystack.includes(normalizedQuery)
    })
  }, [characters, normalizedQuery])

  const groupedCharacters = useMemo(() => {
    const grouped = groupCharactersByLetter(filteredCharacters)
    Object.keys(grouped).forEach((letter) => {
      grouped[letter] = grouped[letter].slice().sort((a, b) => {
        return a.name.localeCompare(b.name)
      })
    })
    return grouped
  }, [filteredCharacters])

  const lettersWithCharacters = useMemo(
    () => alphabet.filter((letter) => groupedCharacters[letter]?.length),
    [groupedCharacters]
  )

  const jumpToLetter = (letter: string) => {
    const target = document.getElementById(
      `character-letter-${getLetterAnchor(letter)}`
    )
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const handleItemKeyDown = (
    event: KeyboardEvent<HTMLDivElement>,
    characterId: string
  ) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(characterId)
    }
  }

  return (
    <div className="flex h-full w-full min-h-0 flex-col">
      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[#2d3138] bg-[#171a1f]/75',
          'shadow-[0_20px_40px_rgba(0,0,0,0.35)]'
        )}
      >
        <div className="flex flex-wrap items-center gap-3 border-b border-[#2a2f36] px-4 py-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#7d8792]" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search characters..."
              aria-label="Search characters"
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
            aria-label="Create character"
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
              {lettersWithCharacters.length ? (
                <div className="space-y-8">
                  {lettersWithCharacters.map((letter) => {
                    const items = groupedCharacters[letter] ?? []
                    return (
                      <section
                        key={letter}
                        id={`character-letter-${getLetterAnchor(letter)}`}
                        className="scroll-mt-6 space-y-3"
                      >
                        <div className="text-[11px] font-semibold uppercase tracking-[0.4em] text-[#8b93a0]">
                          {letter}
                        </div>
                        <ItemGroup>
                          {items.map((character, index) => {
                            const isSelected = selectedId === character.id
                            const promptPreview = getPromptPreview(
                              character.systemPrompt
                            )
                            return (
                              <div key={character.id}>
                                <Item
                                  role="button"
                                  tabIndex={0}
                                  aria-pressed={isSelected}
                                  onClick={() => onSelect(character.id)}
                                  onKeyDown={(event) =>
                                    handleItemKeyDown(event, character.id)
                                  }
                                  size="sm"
                                  className={cn(
                                    'min-h-[112px] gap-4 py-2 border-[#2a2f36] bg-[#1a1d22]/80',
                                    'cursor-pointer hover:bg-[#22262d]',
                                    'focus-visible:border-[#4aa3ff] focus-visible:ring-[#4aa3ff]/40',
                                    isSelected
                                      ? 'border-[#4aa3ff]/60 bg-[#1d222b] shadow-[0_0_0_1px_rgba(74,163,255,0.25)]'
                                      : null
                                  )}
                                >
                                  <ItemMedia className="h-24 w-16 overflow-hidden rounded-[10px] border border-[#2b3139] bg-[#101418]">
                                    {character.imageDataUrl ? (
                                      <img
                                        src={character.imageDataUrl}
                                        alt={character.name}
                                        className="h-full w-full object-cover"
                                      />
                                    ) : (
                                      <div className="flex h-full w-full items-center justify-center text-[11px] font-semibold text-[#aeb6c2]">
                                        {getFallbackInitials(character.name)}
                                      </div>
                                    )}
                                  </ItemMedia>
                                  <ItemContent className="gap-2">
                                    <ItemTitle className="w-full">
                                      <span className="text-sm font-semibold text-[#e2e6ea]">
                                        {character.name}
                                      </span>
                                    </ItemTitle>
                                    {promptPreview ? (
                                      <ItemDescription className="text-[#8b93a0]">
                                        {promptPreview}
                                      </ItemDescription>
                                    ) : null}
                                  </ItemContent>
                                  <ItemActions>
                                    <Button
                                      type="button"
                                      variant="outline"
                                      size="sm"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        onChat?.(character.id)
                                      }}
                                      className={cn(
                                        character.isActive
                                          ? 'border-[#22c55e]/60 bg-[#22c55e]/10 text-[#22c55e] hover:border-[#22c55e]/80 hover:bg-[#22c55e]/20 hover:text-[#22c55e]'
                                          : 'border-[#2f353d] bg-[#1e2228] text-[#cbd2da] hover:border-[#3b424c] hover:bg-[#252a32] hover:text-white'
                                      )}
                                    >
                                      <MessageCircle className={cn(
                                        'h-4 w-4',
                                        character.isActive ? 'text-[#22c55e]' : 'text-[#7fd2ff]'
                                      )} />
                                      Chat
                                    </Button>
                                  </ItemActions>
                                </Item>
                                {index < items.length - 1 ? (
                                  <ItemSeparator className="my-2 bg-[#2a2f36]" />
                                ) : null}
                              </div>
                            )
                          })}
                        </ItemGroup>
                      </section>
                    )
                  })}
                </div>
              ) : (
                <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-2 text-center text-sm text-[#7a828c]">
                  {characters.length ? (
                    <span>No characters match your search.</span>
                  ) : (
                    <>
                      <span className="text-[#a5adb8]">No characters yet.</span>
                      <span>Create one to start building.</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </ScrollArea>
          <div className="hidden w-10 flex-col items-center gap-1 border-l border-[#242a31] py-4 pr-3 text-[10px] font-semibold text-[#6f7782] md:flex">
            {alphabet.map((letter) => {
              const isActive = groupedCharacters[letter]?.length
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

export default CharacterDirectory
