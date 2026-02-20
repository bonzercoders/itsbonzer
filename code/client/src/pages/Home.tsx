import type { ComponentType, ReactNode } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  Bold,
  Check,
  ChevronDown,
  ChevronsUpDown,
  Code,
  Highlighter,
  Italic,
  List,
  Mic,
  Paperclip,
  Plus,
  Quote,
  Redo2,
  Send,
  Undo2,
} from 'lucide-react'

import arrowLeft from '@/assets/arrow-left.png'
import arrowRight from '@/assets/arrow-right.png'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import type { Character } from '@/components/characters/types'
import { MessageArea } from '@/components/chat/MessageArea'
import type { ChatMessage } from '@/components/chat/types'
import { fetchCharacters } from '@/lib/api/characters'
import {
  fetchOpenRouterModelGroups,
  type ModelGroup,
} from '@/lib/openrouter-models'
import { cn } from '@/lib/utils'
import {
  WebSocketClient,
  type ConnectionStatus,
  type InboundMessage,
  type ModelSettingsMessage,
} from '@/lib/websocket'
import { AudioPlayer } from '@/lib/audioPlayer'
import { STTAudioCapture } from '@/lib/sttAudioCapture'

type ToolbarButtonProps = {
  label: string
  icon?: ComponentType<{ className?: string }>
  children?: ReactNode
}

function ToolbarButton({ label, icon: Icon, children }: ToolbarButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-7 items-center justify-center gap-1 rounded-md px-2 text-[11px] text-[#a7aeb7]',
        'hover:bg-[#22262d] hover:text-[#e3e7ec]'
      )}
    >
      {Icon ? <Icon className="h-3.5 w-3.5" /> : children}
    </button>
  )
}

type ModelParameterValues = {
  temperature: number
  topP: number
  minP: number
  topK: number
  frequencyPenalty: number
  presencePenalty: number
  repetitionPenalty: number
}

type ModelParameterConfig = {
  key: keyof ModelParameterValues
  label: string
  min: number
  max: number
  step: number
  decimals: number
  sliderClassName: string
}

const defaultModelParameterValues: ModelParameterValues = {
  temperature: 1,
  topP: 1,
  minP: 0,
  topK: 50,
  frequencyPenalty: 0,
  presencePenalty: 0,
  repetitionPenalty: 1,
}

const modelParameterConfigs: ModelParameterConfig[] = [
  {
    key: 'temperature',
    label: 'Temperature',
    min: 0,
    max: 2,
    step: 0.01,
    decimals: 2,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#4cc8ff] [&_[data-slot=slider-thumb]]:border-[#4cc8ff] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(76,200,255,0.22)]',
  },
  {
    key: 'topP',
    label: 'Top P',
    min: 0,
    max: 1,
    step: 0.01,
    decimals: 2,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#4deca2] [&_[data-slot=slider-thumb]]:border-[#4deca2] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(77,236,162,0.2)]',
  },
  {
    key: 'minP',
    label: 'Min P',
    min: 0,
    max: 1,
    step: 0.01,
    decimals: 2,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#ffcf5c] [&_[data-slot=slider-thumb]]:border-[#ffcf5c] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(255,207,92,0.2)]',
  },
  {
    key: 'topK',
    label: 'Top K',
    min: 0,
    max: 100,
    step: 1,
    decimals: 0,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#b39bff] [&_[data-slot=slider-thumb]]:border-[#b39bff] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(179,155,255,0.2)]',
  },
  {
    key: 'frequencyPenalty',
    label: 'Frequency Penalty',
    min: -2,
    max: 2,
    step: 0.1,
    decimals: 1,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#ff7c8e] [&_[data-slot=slider-thumb]]:border-[#ff7c8e] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(255,124,142,0.2)]',
  },
  {
    key: 'presencePenalty',
    label: 'Presence Penalty',
    min: -2,
    max: 2,
    step: 0.1,
    decimals: 1,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#60a5fa] [&_[data-slot=slider-thumb]]:border-[#60a5fa] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(96,165,250,0.2)]',
  },
  {
    key: 'repetitionPenalty',
    label: 'Repetition Penalty',
    min: 0,
    max: 2,
    step: 0.01,
    decimals: 2,
    sliderClassName:
      '[&_[data-slot=slider-range]]:bg-[#93e67b] [&_[data-slot=slider-thumb]]:border-[#93e67b] [&_[data-slot=slider-thumb]]:shadow-[0_0_0_3px_rgba(147,230,123,0.2)]',
  },
]

const DEFAULT_MODEL_ID = 'google/gemini-2.5-flash'
type MicrophoneState = 'off' | 'enabling' | 'on' | 'error'

type ModelParameterSliderProps = {
  config: ModelParameterConfig
  value: number
  onChange: (nextValue: number) => void
}

function ModelParameterSlider({ config, value, onChange }: ModelParameterSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label className="text-xs font-medium text-[#aeb5bf]">{config.label}</label>
        <span className="rounded border border-[#313741] bg-[#1a1e24] px-2 py-0.5 font-mono text-[11px] text-[#d9dee4]">
          {value.toFixed(config.decimals)}
        </span>
      </div>
      <Slider
        min={config.min}
        max={config.max}
        step={config.step}
        value={[value]}
        onValueChange={(sliderValues) => {
          const nextValue = sliderValues[0]
          if (typeof nextValue === 'number') {
            onChange(nextValue)
          }
        }}
        className={cn(
          'h-5 [&_[data-slot=slider-track]]:h-2 [&_[data-slot=slider-track]]:bg-[#28303a] [&_[data-slot=slider-thumb]]:size-5 [&_[data-slot=slider-thumb]]:border-2 [&_[data-slot=slider-thumb]]:bg-[#f8fbff]',
          config.sliderClassName
        )}
      />
    </div>
  )
}

type ModelComboboxProps = {
  value: string
  onChange: (nextValue: string) => void
}

function ModelCombobox({ value, onChange }: ModelComboboxProps) {
  const [open, setOpen] = useState(false)
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const [triggerWidth, setTriggerWidth] = useState<number | null>(null)

  const selected = modelGroups
    .flatMap((group) => group.options)
    .find((option) => option.value === value)

  useEffect(() => {
    let isCancelled = false

    const loadModelGroups = async () => {
      setIsLoading(true)
      setLoadError(null)

      try {
        const groups = await fetchOpenRouterModelGroups()

        if (isCancelled) {
          return
        }

        setModelGroups(groups)
      } catch {
        if (isCancelled) {
          return
        }

        setLoadError('Unable to load models from OpenRouter.')
      } finally {
        if (!isCancelled) {
          setIsLoading(false)
        }
      }
    }

    void loadModelGroups()

    return () => {
      isCancelled = true
    }
  }, [])

  useEffect(() => {
    if (modelGroups.length === 0) {
      return
    }

    const hasCurrentValue = modelGroups.some((group) =>
      group.options.some((option) => option.value === value)
    )
    if (hasCurrentValue) {
      return
    }

    const fallbackValue = modelGroups[0]?.options[0]?.value ?? ''
    if (fallbackValue) {
      onChange(fallbackValue)
    }
  }, [modelGroups, onChange, value])

  useEffect(() => {
    if (!open) {
      return
    }

    const width = triggerRef.current?.offsetWidth ?? null
    setTriggerWidth(width)
  }, [open])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          ref={triggerRef}
          disabled={isLoading && modelGroups.length === 0}
          className={cn(
            'flex min-h-10 w-full items-center justify-between rounded-lg border border-[#2c323a] bg-[#171a1f] px-3.5 py-2.5 text-sm text-[#c9cfd6]',
            'transition-[border-color,color,box-shadow] hover:border-[#77bef554] hover:text-white hover:shadow-[0_0_0_1px_rgba(0,122,204,0.45)]',
            isLoading && modelGroups.length === 0 && 'cursor-not-allowed opacity-60'
          )}
        >
          <span className="truncate text-left">
            {selected?.label ??
              (isLoading
                ? 'Loading models...'
                : loadError
                  ? 'Unable to load models'
                  : 'Select model...')}
          </span>
          <ChevronsUpDown className="h-4 w-4 text-[#7b838d]" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="border border-[#2c323a] bg-[#171a1f] p-0 text-[#d7dce2]"
        align="start"
        style={triggerWidth ? { width: `${triggerWidth}px` } : undefined}
      >
        <Command
          className={cn(
            'bg-transparent text-[#d7dce2]',
            '[&_[data-slot=command-input-wrapper]]:border-[#2c323a]',
            '[&_[data-slot=command-input-wrapper]_svg]:text-[#7b838d]',
            '[&_[data-slot=command-item][data-selected=true]]:bg-[#242a31]',
            '[&_[data-slot=command-item][data-selected=true]]:text-white'
          )}
        >
          <CommandInput
            placeholder="Search models..."
            className="text-sm text-[#d7dce2] placeholder:text-[#7b838d]"
          />
          <CommandList className="model-combobox-list max-h-[280px]">
            <CommandEmpty className="text-[#8b93a0]">
              {isLoading
                ? 'Loading models...'
                : loadError
                  ? loadError
                  : 'No models found.'}
            </CommandEmpty>
            {modelGroups.map((group, index) => (
              <div key={group.label}>
                <CommandGroup heading={group.label}>
                  {group.options.map((option) => (
                    <CommandItem
                      key={option.value}
                      value={option.value}
                      onSelect={(currentValue) => {
                        onChange(currentValue)
                        setOpen(false)
                      }}
                      className="text-sm text-[#c9cfd6] data-[selected=true]:text-white"
                    >
                      {option.label}
                      <Check
                        className={cn(
                          'ml-auto h-4 w-4 text-[#6fc3ff] transition-opacity',
                          value === option.value ? 'opacity-100' : 'opacity-0'
                        )}
                      />
                    </CommandItem>
                  ))}
                </CommandGroup>
                {index < modelGroups.length - 1 ? (
                  <CommandSeparator className="bg-[#242a31]" />
                ) : null}
              </div>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

function mapModelSettingsMessage(
  modelId: string,
  modelParameters: ModelParameterValues
): ModelSettingsMessage {
  return {
    type: 'model_settings',
    model: modelId || DEFAULT_MODEL_ID,
    temperature: modelParameters.temperature,
    top_p: modelParameters.topP,
    min_p: modelParameters.minP,
    top_k: Math.round(modelParameters.topK),
    frequency_penalty: modelParameters.frequencyPenalty,
    presence_penalty: modelParameters.presencePenalty,
    repetition_penalty: modelParameters.repetitionPenalty,
  }
}

function HomePage() {
  const [leftOpen, setLeftOpen] = useState(false)
  const [rightOpen, setRightOpen] = useState(false)
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('disconnected')
  const [messageText, setMessageText] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [modelParameters, setModelParameters] = useState<ModelParameterValues>(
    defaultModelParameterValues
  )
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [microphoneState, setMicrophoneState] = useState<MicrophoneState>('off')
  const [characterMap, setCharacterMap] = useState<Map<string, Character>>(new Map())
  const streamingTextRef = useRef<HTMLDivElement | null>(null)
  const streamingMessageIdRef = useRef<string | null>(null)
  const liveSttMessageIdRef = useRef<string | null>(null)
  const microphoneStateRef = useRef<MicrophoneState>('off')
  const websocketClientRef = useRef<WebSocketClient | null>(null)
  const audioPlayerRef = useRef<AudioPlayer | null>(null)
  const sttAudioCaptureRef = useRef<STTAudioCapture | null>(null)
  const modelSettingsRef = useRef<ModelSettingsMessage>(
    mapModelSettingsMessage(DEFAULT_MODEL_ID, defaultModelParameterValues)
  )
  const drawerWidth = 420

  if (websocketClientRef.current === null) {
    websocketClientRef.current = new WebSocketClient()
  }

  if (audioPlayerRef.current === null) {
    audioPlayerRef.current = new AudioPlayer()
  }
  if (sttAudioCaptureRef.current === null) {
    sttAudioCaptureRef.current = new STTAudioCapture({
      sendBinary: (data) => websocketClientRef.current?.sendBinary(data) ?? false,
    })
  }

  useEffect(() => {
    modelSettingsRef.current = mapModelSettingsMessage(
      selectedModel || DEFAULT_MODEL_ID,
      modelParameters
    )
  }, [modelParameters, selectedModel])

  useEffect(() => {
    microphoneStateRef.current = microphoneState
  }, [microphoneState])

  useEffect(() => {
    let cancelled = false
    fetchCharacters()
      .then((chars) => {
        if (cancelled) return
        const map = new Map<string, Character>()
        for (const c of chars) map.set(c.id, c)
        setCharacterMap(map)
      })
      .catch((err: unknown) => console.error('[Characters] failed to load', err))
    return () => { cancelled = true }
  }, [])

  const upsertLiveSttMessage = useCallback(
    (liveText: string) => {
      setMessages((prev) => {
        const liveId = liveSttMessageIdRef.current

        if (liveId) {
          let found = false
          const next = prev.map((message): ChatMessage => {
            if (message.id !== liveId) {
              return message
            }

            found = true
            const nextRealtime = { liveText }

            return {
              ...message,
              isStreaming: true,
              content: nextRealtime.liveText,
              realtime: nextRealtime,
            }
          })

          if (found) {
            return next
          }
        }

        const lastMessage = prev[prev.length - 1]
        if (
          lastMessage &&
          lastMessage.role === 'user' &&
          lastMessage.inputMode === 'stt'
        ) {
          liveSttMessageIdRef.current = lastMessage.id

          return prev.map((message): ChatMessage => {
            if (message.id !== lastMessage.id) {
              return message
            }

            return {
              ...message,
              isStreaming: true,
              content: liveText,
              realtime: { liveText },
            }
          })
        }

        const id = crypto.randomUUID()
        liveSttMessageIdRef.current = id

        const realtime = { liveText }
        const liveMessage: ChatMessage = {
          id,
          role: 'user',
          inputMode: 'stt',
          name: null,
          characterId: null,
          content: realtime.liveText,
          isStreaming: true,
          realtime,
        }

        return [
          ...prev,
          liveMessage,
        ]
      })
    },
    []
  )

  const finalizeLiveSttMessage = useCallback((finalText: string) => {
    setMessages((prev) => {
      const trimmedFinal = finalText.trim()
      const liveId = liveSttMessageIdRef.current
      liveSttMessageIdRef.current = null

      if (liveId) {
        let found = false
        const next = prev.map((message): ChatMessage => {
          if (message.id !== liveId) {
            return message
          }

          found = true
          return {
            ...message,
            inputMode: 'stt',
            content: trimmedFinal,
            isStreaming: false,
            realtime: undefined,
          }
        })

        if (found) {
          if (trimmedFinal) {
            return next
          }
          return next.filter((message) => message.id !== liveId)
        }
      }

      const lastMessage = prev[prev.length - 1]
      if (
        lastMessage &&
        lastMessage.role === 'user' &&
        lastMessage.inputMode === 'stt'
      ) {
        if (!trimmedFinal) {
          return prev
        }

        return prev.map((message): ChatMessage => {
          if (message.id !== lastMessage.id) {
            return message
          }

          return {
            ...message,
            content: trimmedFinal,
            isStreaming: false,
            realtime: undefined,
          }
        })
      }

      if (!trimmedFinal) {
        return prev
      }
      const finalMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        inputMode: 'stt',
        name: null,
        characterId: null,
        content: trimmedFinal,
        isStreaming: false,
      }

      return [
        ...prev,
        finalMessage,
      ]
    })
  }, [])

  const sealLiveSttMessage = useCallback(() => {
    setMessages((prev) => {
      const liveId = liveSttMessageIdRef.current
      if (!liveId) {
        return prev
      }

      liveSttMessageIdRef.current = null

      return prev.flatMap((message): ChatMessage[] => {
        if (message.id !== liveId) {
          return [message]
        }

        const fallbackText = message.realtime?.liveText || message.content

        if (!fallbackText.trim()) {
          return []
        }

        const sealedMessage: ChatMessage = {
          ...message,
          inputMode: 'stt',
          content: fallbackText,
          isStreaming: false,
          realtime: undefined,
        }

        return [sealedMessage]
      })
    })
  }, [])

  const handleWebSocketMessage = useCallback((message: InboundMessage) => {
    if (message.type === 'text_stream_start') {
      const { character_id, character_name, message_id } = message.data
      streamingMessageIdRef.current = message_id
      setMessages((prev) => [
        ...prev,
        {
          id: message_id,
          role: 'assistant',
          name: character_name,
          characterId: character_id,
          content: '',
          isStreaming: true,
        },
      ])
    } else if (message.type === 'text_chunk') {
      if (message.data.message_id !== streamingMessageIdRef.current) return
      if (streamingTextRef.current) {
        streamingTextRef.current.textContent += message.data.text
      }
    } else if (message.type === 'text_stream_stop') {
      const { message_id, text } = message.data
      streamingMessageIdRef.current = null
      if (!text.trim()) {
        setMessages((prev) => prev.filter((m) => m.id !== message_id))
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === message_id ? { ...m, content: text, isStreaming: false } : m
          )
        )
      }
    } else if (message.type === 'stt_update') {
      upsertLiveSttMessage(message.text)
    } else if (message.type === 'stt_stabilized') {
      // `stt_update` is the primary live stream for word-by-word display.
      // Keep stabilized callbacks passive to avoid visual rewinds/flicker.
      return
    } else if (message.type === 'stt_final') {
      finalizeLiveSttMessage(message.text)
    } else if (message.type === 'audio_stream_start') {
      sttAudioCaptureRef.current?.setPausedForPlayback(true)
      audioPlayerRef.current?.handleStreamStart(message.data)
    } else if (message.type === 'audio_stream_stop') {
      sttAudioCaptureRef.current?.setPausedForPlayback(false)
      audioPlayerRef.current?.handleStreamStop(message.data)
    }
  }, [finalizeLiveSttMessage, upsertLiveSttMessage])

  useEffect(() => {
    const websocketClient = websocketClientRef.current
    if (!websocketClient) {
      return
    }

    setConnectionStatus(websocketClient.getStatus())

    const unsubscribeStatus = websocketClient.onStatusChange((status) => {
      setConnectionStatus(status)

      if (status === 'connected') {
        websocketClient.send(modelSettingsRef.current)
        if (microphoneStateRef.current === 'on') {
          websocketClient.send({ type: 'start_listening' })
        }
      }

      // Finalize any in-flight streaming message on disconnect
      if (status === 'disconnected' || status === 'reconnecting') {
        sttAudioCaptureRef.current?.setPausedForPlayback(false)

        const activeId = streamingMessageIdRef.current
        if (activeId) {
          streamingMessageIdRef.current = null
          const partialText = streamingTextRef.current?.textContent ?? ''
          setMessages((prev) =>
            prev.map((m) =>
              m.id === activeId
                ? { ...m, content: partialText || m.content, isStreaming: false }
                : m
            )
          )
        }

        sealLiveSttMessage()
      }
    })

    const unsubscribeMessage = websocketClient.onMessage(handleWebSocketMessage)

    const unsubscribeBinary = websocketClient.onBinary((audioBuffer) => {
      audioPlayerRef.current?.handleAudioChunk(audioBuffer)
    })

    const unsubscribeError = websocketClient.onError((error) => {
      console.error('[WS] error', error)
    })

    void websocketClient.connect().catch((error: unknown) => {
      console.error('[WS] connect failed', error)
    })

    return () => {
      if (microphoneStateRef.current === 'on') {
        websocketClient.send({ type: 'stop_listening' })
      }
      sttAudioCaptureRef.current?.disable()
      void sttAudioCaptureRef.current?.destroy()
      sttAudioCaptureRef.current = null

      unsubscribeStatus()
      unsubscribeMessage()
      unsubscribeBinary()
      unsubscribeError()
      websocketClient.disconnect()
      audioPlayerRef.current?.destroy()
      audioPlayerRef.current = null
    }
  }, [handleWebSocketMessage, sealLiveSttMessage])

  const toggleMicrophone = useCallback(async () => {
    const websocketClient = websocketClientRef.current
    const sttAudioCapture = sttAudioCaptureRef.current

    if (!websocketClient || !sttAudioCapture) {
      return
    }

    if (microphoneStateRef.current === 'enabling') {
      return
    }

    if (microphoneStateRef.current === 'on') {
      websocketClient.send({ type: 'stop_listening' })
      sttAudioCapture.disable()
      setMicrophoneState('off')
      sealLiveSttMessage()
      return
    }

    if (websocketClient.getStatus() !== 'connected') {
      setMicrophoneState('error')
      console.error('[STT] Cannot start microphone while websocket is disconnected')
      return
    }

    setMicrophoneState('enabling')

    const enabled = await sttAudioCapture.enable()
    if (!enabled) {
      setMicrophoneState('error')
      console.error('[STT] Failed to enable audio capture', sttAudioCapture.getState())
      return
    }

    const started = websocketClient.send({ type: 'start_listening' })
    if (!started) {
      sttAudioCapture.disable()
      setMicrophoneState('error')
      console.error('[STT] Failed to send start_listening command')
      return
    }

    sttAudioCapture.setPausedForPlayback(false)
    setMicrophoneState('on')
  }, [sealLiveSttMessage])

  const sendUserMessage = () => {
    const websocketClient = websocketClientRef.current
    const text = messageText.trim()
    if (!websocketClient || !text) {
      return
    }

    const modelSettingsSent = websocketClient.send(modelSettingsRef.current)
    const userMessageSent = websocketClient.send({ type: 'user_message', text })

    if (modelSettingsSent && userMessageSent) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'user',
          inputMode: 'typed',
          name: null,
          characterId: null,
          content: text,
          isStreaming: false,
        },
      ])
      setMessageText('')
    }
  }

  const connectionLabel =
    connectionStatus === 'connected'
      ? 'connected'
      : connectionStatus === 'disconnected'
        ? 'disconnected'
        : 'connecting'

  const connectionColorClass =
    connectionLabel === 'connected'
      ? 'text-[#22c55e]'
      : connectionLabel === 'connecting'
        ? 'text-[#f59e0b]'
        : 'text-[#9ca3af]'

  const connectionDotClass =
    connectionLabel === 'connected'
      ? 'border-[#16a34a] bg-[#22c55e]'
      : connectionLabel === 'connecting'
        ? 'border-[#d97706] bg-[#f59e0b]'
        : 'border-[#4b5563] bg-[#6b7280]'

  const microphoneButtonClass = cn(
    'inline-flex h-9 w-9 items-center justify-center rounded-full border text-[#c8cdd3]',
    microphoneState === 'on'
      ? 'border-[#0d7bc6] bg-[#007acc] text-white hover:bg-[#1087d9]'
      : microphoneState === 'enabling'
        ? 'border-[#4b5563] bg-[#2a2f36] text-[#9ca3af]'
        : microphoneState === 'error'
          ? 'border-[#b91c1c] bg-[#3a1f22] text-[#fecaca] hover:border-[#ef4444]'
          : 'border-[#2f353d] bg-[#22252a] hover:border-[#3f4650] hover:text-white'
  )

  const microphoneLabel =
    microphoneState === 'on'
      ? 'Microphone on'
      : microphoneState === 'enabling'
        ? 'Enabling microphone'
        : microphoneState === 'error'
          ? 'Microphone error'
          : 'Microphone off'

  return (
    <div className="relative flex h-full w-full overflow-hidden">
      <div
        className="pointer-events-none absolute inset-y-0 left-0 z-10 flex"
        style={{ width: drawerWidth }}
      >
        <div
          className={cn(
            'pointer-events-auto h-full w-full rounded-2xl border border-[#2d3138] bg-[#191c21]',
            leftOpen && 'shadow-[0_20px_45px_rgba(0,0,0,0.4)]',
            'transition-transform duration-300 ease-out'
          )}
          style={{
            transform: `translateX(${leftOpen ? 0 : -drawerWidth}px)`,
          }}
        >
          <div className="flex h-full flex-col gap-4 p-6 text-sm text-[#cbd1d8]">
            <div className="text-base font-semibold text-white">Info Panel</div>
            <div className="h-px w-full bg-[#2a2f36]" />
            <div className="text-xs text-[#9aa2ab]">
              Placeholder content for the left drawer.
            </div>
          </div>
        </div>
        <button
          type="button"
          aria-label={leftOpen ? 'Collapse left drawer' : 'Expand left drawer'}
          onClick={() => setLeftOpen((value) => !value)}
          className="pointer-events-auto absolute left-0 top-1/2 z-20 transition-transform duration-300 ease-out"
          style={{
            transform: `translate(${leftOpen ? drawerWidth : 0}px, -50%)`,
          }}
        >
          <img
            src={arrowRight}
            alt=""
            className={cn(
              'h-8 w-8 transition-transform duration-300 ease-out',
              leftOpen && 'rotate-180'
            )}
          />
        </button>
      </div>

      <div className="flex-1 h-full overflow-hidden">
        <MessageArea
          messages={messages}
          characterMap={characterMap}
          streamingRef={streamingTextRef}
        />
      </div>

      <div
        className="pointer-events-none absolute inset-y-0 right-0 z-10 flex justify-end"
        style={{ width: drawerWidth }}
      >
        <div
          className={cn(
            'pointer-events-auto h-full w-full rounded-2xl border border-[#2d3138] bg-[#191c21]',
            rightOpen && 'shadow-[0_20px_45px_rgba(0,0,0,0.4)]',
            'transition-transform duration-300 ease-out'
          )}
          style={{
            transform: `translateX(${rightOpen ? 0 : drawerWidth}px)`,
          }}
        >
          <div className="flex h-full flex-col gap-4 overflow-y-auto p-6 text-sm text-[#cbd1d8]">
            <div className="text-base font-semibold text-white">
              Model Settings
            </div>
            <div className="h-px w-full bg-[#2a2f36]" />
            <div className="space-y-6">
              <div className="space-y-3">
                <div className="text-xs font-semibold uppercase tracking-[0.2em] text-[#8b929b]">
                  Model
                </div>
                <ModelCombobox value={selectedModel} onChange={setSelectedModel} />
              </div>
              <div className="space-y-4">
                <div className="text-xs font-semibold uppercase tracking-[0.2em] text-[#8b929b]">
                  Parameters
                </div>
                <div className="space-y-4">
                  {modelParameterConfigs.map((config) => (
                    <ModelParameterSlider
                      key={config.key}
                      config={config}
                      value={modelParameters[config.key]}
                      onChange={(nextValue) => {
                        setModelParameters((currentValues) => ({
                          ...currentValues,
                          [config.key]: nextValue,
                        }))
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
        <button
          type="button"
          aria-label={rightOpen ? 'Collapse right drawer' : 'Expand right drawer'}
          onClick={() => setRightOpen((value) => !value)}
          className="pointer-events-auto absolute right-0 top-1/2 z-20 transition-transform duration-300 ease-out"
          style={{
            transform: `translate(${rightOpen ? -drawerWidth : 0}px, -50%)`,
          }}
        >
          <img
            src={arrowLeft}
            alt=""
            className={cn(
              'h-8 w-8 transition-transform duration-300 ease-out',
              rightOpen && 'rotate-180'
            )}
          />
        </button>
      </div>

      <div className="absolute bottom-6 left-1/2 w-[min(860px,92%)] -translate-x-1/2">
        <div className="overflow-hidden rounded-2xl border border-[#333] bg-[#1b1e23] shadow-[0_20px_50px_rgba(0,0,0,0.55)]">
          <div className="flex items-center gap-1 border-b border-[#333] px-3 py-2">
            <ToolbarButton label="Undo" icon={Undo2} />
            <ToolbarButton label="Redo" icon={Redo2} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <div className="relative">
              <select
                className={cn(
                  'h-7 w-[110px] appearance-none rounded-md border border-transparent bg-transparent px-2 text-[11px] text-[#a7aeb7]',
                  'hover:bg-[#22262d] hover:text-[#e3e7ec] focus:border-[#3c424b] focus:outline-none'
                )}
                defaultValue="Paragraph"
                aria-label="Text style"
              >
                <option>Paragraph</option>
                <option>H1</option>
                <option>H2</option>
                <option>H3</option>
                <option>H4</option>
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-[#6f7782]" />
            </div>
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="List" icon={List} />
            <ToolbarButton label="Quote" icon={Quote} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="Bold" icon={Bold} />
            <ToolbarButton label="Italic" icon={Italic} />
            <ToolbarButton label="Code" icon={Code} />
            <ToolbarButton label="Highlight" icon={Highlighter} />
            <div className="h-4 w-px bg-[#2f343b]" />
            <ToolbarButton label="Align left" icon={AlignLeft} />
            <ToolbarButton label="Align center" icon={AlignCenter} />
            <ToolbarButton label="Align right" icon={AlignRight} />
            <div className="ml-auto flex items-center gap-2">
              <span
                className={cn(
                  'text-[11px] font-medium lowercase tracking-[0.02em]',
                  connectionColorClass
                )}
              >
                {connectionLabel}
              </span>
              <span
                className={cn('h-2.5 w-2.5 rounded-full border', connectionDotClass)}
                aria-label="Connection status"
                title="Connection status"
              />
            </div>
          </div>

          <textarea
            className={cn(
              'min-h-[140px] w-full resize-none bg-transparent px-4 py-3 text-sm text-[#dfe3e8] outline-none',
              'placeholder:text-[#6c7480]'
            )}
            placeholder="Type your message..."
            value={messageText}
            onChange={(event) => setMessageText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                sendUserMessage()
              }
            }}
          />

          <div className="flex items-center justify-between px-3 py-3">
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#2f353d] bg-[#22252a] text-[#c8cdd3] hover:border-[#3f4650] hover:text-white"
                aria-label="Attach file"
                title="Attach file"
              >
                <Paperclip className="h-4 w-4 text-[#7fd2ff]" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#2f353d] bg-[#22252a] text-[#c8cdd3] hover:border-[#3f4650] hover:text-white"
                aria-label="Add options"
                title="Add options"
              >
                <Plus className="h-4 w-4" />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className={microphoneButtonClass}
                aria-label={microphoneLabel}
                title={microphoneLabel}
                onClick={() => {
                  void toggleMicrophone()
                }}
                disabled={microphoneState === 'enabling'}
              >
                <Mic
                  className={cn(
                    'h-4 w-4',
                    microphoneState === 'enabling' && 'animate-pulse'
                  )}
                />
              </button>
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-lg bg-[#007acc] px-4 py-2 text-xs font-semibold text-white shadow-[0_8px_18px_rgba(0,122,204,0.35)] hover:bg-[#1087d9]"
                onClick={sendUserMessage}
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HomePage
