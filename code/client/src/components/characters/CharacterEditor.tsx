import { ChevronDown, X } from 'lucide-react'
import { type ChangeEvent, useState } from 'react'

import { type Character } from '@/components/characters/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

type VoiceOption = {
  value: string
  label: string
}

type CharacterEditorProps = {
  character: Character
  voiceOptions?: VoiceOption[]
  onChat?: () => void
  onChange: (updates: Partial<Character>) => void
  onClose?: () => void
  onDelete?: () => void
  onSave?: () => void
}

function CharacterEditor({
  character,
  voiceOptions = [],
  onChat,
  onChange,
  onClose,
  onDelete,
  onSave,
}: CharacterEditorProps) {
  const [isGlobalPromptOpen, setIsGlobalPromptOpen] = useState(false)
  const displayName = character.name.trim() || 'Character Name'
  const isPlaceholderName = !character.name.trim()
  const inputClassName = cn(
    'h-10 border-[#2d3138] bg-[#171a1f] text-sm text-[#dfe3e8] shadow-none',
    'placeholder:text-[#6c7480] focus-visible:border-[#3a414b] focus-visible:ring-[#2c323a]'
  )
  const textareaClassName = cn(
    'border-[#2d3138] bg-[#171a1f] text-sm text-[#dfe3e8] shadow-none',
    'placeholder:text-[#6c7480] focus-visible:border-[#3a414b] focus-visible:ring-[#2c323a]',
    '[field-sizing:fixed] resize-y overflow-y-auto'
  )
  const labelClassName = 'text-sm font-medium text-[#cbd2da]'
  const actionButtonClassName = 'h-10 justify-center px-6 text-sm font-semibold'
  const promptWidthClassName = 'w-full max-w-[640px]'

  const handleMainImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]

    if (!file) {
      onChange({ imageDataUrl: undefined })
      return
    }

    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onChange({ imageDataUrl: reader.result })
      }
    }
    reader.readAsDataURL(file)
  }

  return (
    <div className="flex h-full w-full flex-col rounded-2xl border border-[#2b3139] bg-[#171a1f]/75 p-6 pb-10 shadow-[0_20px_40px_rgba(0,0,0,0.35)]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div
          className={cn(
            'text-2xl font-semibold tracking-tight md:text-3xl',
            isPlaceholderName ? 'text-[#8b929b]' : 'text-[#e5ebf3]'
          )}
        >
          {displayName}
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="text-[#9aa2ad] hover:bg-[#242a31] hover:text-white"
          onClick={onClose}
          disabled={!onClose}
          aria-label="Close editor"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      <Tabs defaultValue="profile" className="mt-6 flex h-full flex-col">
        <TabsList className="h-auto w-full justify-start gap-6 rounded-none border-b border-[#2a2f36] bg-transparent p-0">
          {['Profile', 'Persona', 'Chats', 'Groups', 'Memory'].map((label) => {
            const value = label.toLowerCase()
            return (
              <TabsTrigger
                key={value}
                value={value}
                className={cn(
                  'relative flex-none rounded-none border-0 bg-transparent px-0 pb-3 pt-0 text-sm font-medium text-[#8c939d] shadow-none',
                  'h-auto',
                  "after:absolute after:bottom-0 after:left-0 after:h-[2px] after:w-full after:rounded-full after:bg-transparent after:content-['']",
                  'data-[state=active]:text-[#e5ebf3] data-[state=active]:shadow-none data-[state=active]:after:bg-[#4aa3ff]'
                )}
              >
                {label}
              </TabsTrigger>
            )
          })}
        </TabsList>

        <TabsContent value="profile" className="mt-6 flex-1">
          <div className="flex w-full flex-col gap-6">
            <div className="grid w-full items-start gap-8 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
              <div className="flex w-full flex-col gap-4">
                <label
                  htmlFor="character-main-image"
                  className={cn(
                    'group relative flex aspect-[4/5] w-full max-w-[320px] cursor-pointer items-center justify-center overflow-hidden rounded-2xl border border-[#2b3139] bg-[#14171b]',
                    'transition-colors hover:border-[#3a414b]'
                  )}
                >
                  {character.imageDataUrl ? (
                    <img
                      src={character.imageDataUrl}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <span className="sr-only">Upload image</span>
                  )}
                </label>
                <input
                  id="character-main-image"
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={handleMainImageChange}
                />
              </div>

              <div className="flex flex-col gap-5 lg:h-[400px] lg:justify-between">
                <div className="flex flex-col gap-4">
                  <div className={cn(promptWidthClassName, 'space-y-2')}>
                    <button
                      type="button"
                      className={cn(
                        labelClassName,
                        'flex w-full items-center justify-between gap-2 text-left'
                      )}
                      onClick={() =>
                        setIsGlobalPromptOpen((previous) => !previous)
                      }
                      aria-expanded={isGlobalPromptOpen}
                      aria-controls="global-system-prompt-panel"
                    >
                      <span>Global Roleplay System Prompt</span>
                      <ChevronDown
                        className={cn(
                          'h-4 w-4 text-[#7c8794] transition-transform duration-300',
                          isGlobalPromptOpen ? 'rotate-180' : ''
                        )}
                      />
                    </button>
                    <div
                      id="global-system-prompt-panel"
                      className={cn(
                        'overflow-hidden transition-[max-height,opacity] duration-300 ease-out',
                        isGlobalPromptOpen
                          ? 'max-h-[200px] opacity-100'
                          : 'max-h-0 opacity-0'
                      )}
                    >
                      <div className="pt-3">
                        <Textarea
                          id="global-system-prompt"
                          rows={3}
                          aria-label="Global Roleplay System Prompt"
                          value={character.globalPrompt}
                          onChange={(event) =>
                            onChange({ globalPrompt: event.target.value })
                          }
                          className={cn(
                            textareaClassName,
                            'h-[84px] min-h-[84px]'
                          )}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="w-full max-w-[360px] space-y-2">
                    <Label htmlFor="character-name" className={labelClassName}>
                      Character Name
                    </Label>
                    <Input
                      id="character-name"
                      value={character.name}
                      onChange={(event) =>
                        onChange({ name: event.target.value })
                      }
                      className={inputClassName}
                    />
                  </div>

                  <div className="w-full max-w-[360px] space-y-2">
                    <Label className={labelClassName}>Voice</Label>
                    <Select
                      value={character.voiceId}
                      onValueChange={(value) => onChange({ voiceId: value })}
                    >
                      <SelectTrigger className={cn(inputClassName, 'w-full')}>
                        <SelectValue placeholder="Select voice" />
                      </SelectTrigger>
                      <SelectContent>
                        {voiceOptions.map((voiceOption) => (
                          <SelectItem key={voiceOption.value} value={voiceOption.value}>
                            {voiceOption.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className={cn(promptWidthClassName, 'space-y-2')}>
                  <Label className={labelClassName}>System Prompt</Label>
                  <Textarea
                    rows={4}
                    value={character.systemPrompt}
                    onChange={(event) =>
                      onChange({ systemPrompt: event.target.value })
                    }
                    className={cn(
                      textareaClassName,
                      'h-[120px] min-h-[120px]'
                    )}
                  />
                </div>
              </div>
            </div>

            <div className="grid w-full items-start gap-8 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)] lg:pt-2">
              <div className="flex items-start">
                <Button
                  variant="outline"
                  className={cn(
                    actionButtonClassName,
                    'border-[#6f2c2c] bg-transparent text-[#f1b6b6] hover:bg-[#352022] hover:text-white'
                  )}
                  type="button"
                  onClick={onDelete}
                  disabled={!onDelete}
                >
                  Delete
                </Button>
              </div>
              <div className="flex items-start justify-end gap-3">
                <Button
                  variant="outline"
                  className={cn(
                    actionButtonClassName,
                    character.isActive
                      ? 'border-[#22c55e]/60 bg-[#22c55e]/10 text-[#22c55e] hover:border-[#22c55e]/80 hover:bg-[#22c55e]/20 hover:text-[#22c55e]'
                      : 'border-[#2f353d] bg-[#1e2228] text-[#cbd2da] hover:border-[#3b424c] hover:bg-[#252a32] hover:text-white'
                  )}
                  type="button"
                  onClick={onChat}
                  disabled={!onChat}
                >
                  Chat
                </Button>
                <Button
                  className={cn(
                    actionButtonClassName,
                    'bg-[#1c7fdf] text-white hover:bg-[#2b8eef]'
                  )}
                  type="button"
                  onClick={onSave}
                  disabled={!onSave}
                >
                  Save
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="persona" className="flex-1" />
        <TabsContent value="chats" className="flex-1" />
        <TabsContent value="groups" className="flex-1" />
        <TabsContent value="memory" className="flex-1" />
      </Tabs>
    </div>
  )
}

export default CharacterEditor
