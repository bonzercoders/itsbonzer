import { type FormEvent } from 'react'

import { type Voice, type VoiceMethod } from '@/components/speech/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

type VoiceBuilderFormProps = {
  voiceDraft: Voice
  onChange: (updates: Partial<Voice>) => void
  onSave?: () => void
  onDelete?: () => void
}

const methodOptions: { value: VoiceMethod; title: string }[] = [
  { value: 'clone', title: 'Clone' },
  { value: 'profile', title: 'Profile' },
]

function VoiceBuilderForm({
  voiceDraft,
  onChange,
  onSave,
  onDelete,
}: VoiceBuilderFormProps) {
  const inputClassName = cn(
    'h-10 border-[#2d3138] bg-[#171a1f] text-sm text-[#dfe3e8]',
    'placeholder:text-[#6c7480] focus-visible:border-[#3a414b] focus-visible:ring-[#2c323a]'
  )

  const textareaClassName = cn(
    'h-[155px] min-h-[155px] border-[#2d3138] bg-[#171a1f] text-sm text-[#dfe3e8]',
    'placeholder:text-[#6c7480] focus-visible:border-[#3a414b] focus-visible:ring-[#2c323a]'
  )

  const labelClassName = 'text-sm font-medium text-[#cbd2da]'

  const handleMethodChange = (value: string) => {
    if (value === 'clone' || value === 'profile') {
      onChange({ method: value })
    }
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onSave?.()
  }

  return (
    <div
      className={cn(
        'flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[#2d3138] bg-[#171a1f]/85',
        'shadow-[0_20px_40px_rgba(0,0,0,0.35)]'
      )}
    >
      <form className="flex min-h-0 flex-1 flex-col" onSubmit={handleSubmit}>
        <ScrollArea className="flex-1">
          <div className="space-y-4 px-5 py-4">
            <div className="space-y-2">
              <Label htmlFor="voice-name" className={labelClassName}>
                Voice Name
              </Label>
              <Input
                id="voice-name"
                placeholder="Enter voice name"
                value={voiceDraft.voiceName}
                onChange={(event) => onChange({ voiceName: event.target.value })}
                className={inputClassName}
              />
            </div>

            <div className="space-y-3">
              <Label className={labelClassName}>Method</Label>
              <RadioGroup
                value={voiceDraft.method}
                onValueChange={handleMethodChange}
                className="grid gap-3 sm:grid-cols-2"
              >
                {methodOptions.map((option) => (
                  <Label
                    key={option.value}
                    className={cn(
                      'flex cursor-pointer items-start gap-3 rounded-xl border border-[#2a2f36] bg-[#1a1d22]/80 p-3',
                      'transition hover:border-[#3b424c] hover:bg-[#22262d]',
                      'has-[[data-state=checked]]:border-[#2f6fa3] has-[[data-state=checked]]:bg-[#182634]'
                    )}
                  >
                    <RadioGroupItem
                      id={`method-${option.value}`}
                      value={option.value}
                      className="mt-1 border-[#3a414b] text-[#7fd2ff]"
                    />
                    <div className="space-y-1">
                      <div className="text-sm font-semibold text-[#e2e6ea]">
                        {option.title}
                      </div>
                    </div>
                  </Label>
                ))}
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label htmlFor="scene-prompt" className={labelClassName}>
                Scene Prompt
              </Label>
              <Textarea
                id="scene-prompt"
                placeholder="Enter scene prompt"
                value={voiceDraft.scenePrompt}
                onChange={(event) =>
                  onChange({ scenePrompt: event.target.value })
                }
                className={textareaClassName}
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="reference-text" className={labelClassName}>
                  Reference Text
                </Label>
                <Input
                  id="reference-text"
                  placeholder="Enter reference text path"
                  value={voiceDraft.referenceText}
                  onChange={(event) =>
                    onChange({ referenceText: event.target.value })
                  }
                  className={inputClassName}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="reference-audio" className={labelClassName}>
                  Reference Audio
                </Label>
                <Input
                  id="reference-audio"
                  placeholder="Enter reference audio path"
                  value={voiceDraft.referenceAudio}
                  onChange={(event) =>
                    onChange({ referenceAudio: event.target.value })
                  }
                  className={inputClassName}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="speaker-description" className={labelClassName}>
                Speaker Description
              </Label>
              <Textarea
                id="speaker-description"
                placeholder="Enter speaker description"
                value={voiceDraft.speakerDescription}
                onChange={(event) =>
                  onChange({ speakerDescription: event.target.value })
                }
                className={textareaClassName}
              />
            </div>
          </div>
        </ScrollArea>

        <div className="px-5 py-3">
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              type="submit"
              className="bg-[#007acc] text-white shadow-[0_8px_18px_rgba(0,122,204,0.35)] hover:bg-[#1087d9]"
              disabled={!onSave}
            >
              Save
            </Button>
            <Button
              type="button"
              variant="outline"
              className={cn(
                'border-[#3a2a2a] bg-[#1f1416] text-[#ff8f8f]',
                'hover:border-[#553333] hover:bg-[#2a1618] hover:text-[#ffd2d2]'
              )}
              onClick={onDelete}
              disabled={!onDelete}
            >
              Delete
            </Button>
          </div>
        </div>
      </form>
    </div>
  )
}

export default VoiceBuilderForm
