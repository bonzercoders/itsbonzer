export type STTAudioCaptureStatus = 'idle' | 'enabling' | 'enabled' | 'error'

type STTAudioCaptureOptions = {
  sendBinary: (data: ArrayBuffer) => boolean
  workletModulePath?: string
  targetSampleRate?: number
  frameDurationMs?: number
}

type STTAudioCaptureState = {
  status: STTAudioCaptureStatus
  audioContextState: AudioContextState | null
  error: string | null
}

const DEFAULT_WORKLET_MODULE = '/worklets/stt-capture-processor.js'
const DEFAULT_TARGET_SAMPLE_RATE = 16000
const DEFAULT_FRAME_DURATION_MS = 20

export class STTAudioCapture {
  private readonly sendBinary: (data: ArrayBuffer) => boolean
  private readonly workletModulePath: string
  private readonly targetSampleRate: number
  private readonly frameDurationMs: number

  private context: AudioContext | null = null
  private mediaStream: MediaStream | null = null
  private sourceNode: MediaStreamAudioSourceNode | null = null
  private workletNode: AudioWorkletNode | null = null
  private silentGainNode: GainNode | null = null
  private workletLoaded = false

  private status: STTAudioCaptureStatus = 'idle'
  private errorMessage: string | null = null

  constructor(options: STTAudioCaptureOptions) {
    this.sendBinary = options.sendBinary
    this.workletModulePath = options.workletModulePath ?? DEFAULT_WORKLET_MODULE
    this.targetSampleRate = options.targetSampleRate ?? DEFAULT_TARGET_SAMPLE_RATE
    this.frameDurationMs = options.frameDurationMs ?? DEFAULT_FRAME_DURATION_MS
  }

  async enable(): Promise<boolean> {
    if (this.status === 'enabled') {
      return true
    }
    if (this.status === 'enabling') {
      return false
    }

    this.status = 'enabling'
    this.errorMessage = null

    try {
      await this.ensureAudioContext()
      const context = this.context
      if (!context) {
        throw new Error('AudioContext is unavailable.')
      }

      if (context.state === 'suspended') {
        await context.resume()
      }

      if (!this.workletLoaded) {
        await this.loadWorkletModule(context)
        this.workletLoaded = true
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia is not available in this browser.')
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      this.sourceNode = context.createMediaStreamSource(this.mediaStream)
      this.workletNode = new AudioWorkletNode(context, 'stt-capture-processor', {
        processorOptions: {
          targetSampleRate: this.targetSampleRate,
          frameDurationMs: this.frameDurationMs,
        },
      })
      this.silentGainNode = context.createGain()
      this.silentGainNode.gain.value = 0

      this.workletNode.port.onmessage = (event) => {
        if (this.status !== 'enabled') {
          return
        }

        if (event.data instanceof ArrayBuffer) {
          this.sendBinary(event.data)
        }
      }

      this.sourceNode.connect(this.workletNode)
      this.workletNode.connect(this.silentGainNode)
      this.silentGainNode.connect(context.destination)

      this.status = 'enabled'
      return true
    } catch (error) {
      this.errorMessage =
        error instanceof Error ? error.message : 'Unable to start microphone capture.'
      this.status = 'error'
      this.teardownGraph()
      return false
    }
  }

  disable(): void {
    this.teardownGraph()
    if (this.status !== 'error') {
      this.errorMessage = null
    }
    this.status = 'idle'
  }

  async destroy(): Promise<void> {
    this.disable()

    if (this.context && this.context.state !== 'closed') {
      await this.context.close()
    }

    this.context = null
    this.workletLoaded = false
  }

  getState(): STTAudioCaptureState {
    return {
      status: this.status,
      audioContextState: this.context?.state ?? null,
      error: this.errorMessage,
    }
  }

  private async ensureAudioContext(): Promise<void> {
    if (this.context && this.context.state !== 'closed') {
      return
    }

    if (typeof AudioContext === 'undefined') {
      throw new Error('AudioContext is not supported in this browser.')
    }

    this.context = new AudioContext({ sampleRate: 48000 })
  }

  private async loadWorkletModule(context: AudioContext): Promise<void> {
    try {
      await context.audioWorklet.addModule(this.workletModulePath)
      return
    } catch (primaryError) {
      if (!this.workletModulePath.startsWith('/worklets/')) {
        throw primaryError
      }
    }

    const fallbackPath = `/public${this.workletModulePath}`
    await context.audioWorklet.addModule(fallbackPath)
  }

  private teardownGraph(): void {
    if (this.sourceNode) {
      this.sourceNode.disconnect()
      this.sourceNode = null
    }

    if (this.workletNode) {
      this.workletNode.port.onmessage = null
      this.workletNode.disconnect()
      this.workletNode = null
    }

    if (this.silentGainNode) {
      this.silentGainNode.disconnect()
      this.silentGainNode = null
    }

    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop()
      }
      this.mediaStream = null
    }
  }
}
