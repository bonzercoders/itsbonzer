export interface AudioSession {
  characterId: string
  characterName: string
  messageId: string
  sampleRate: number
  pendingChunks: ArrayBuffer[]
  done: boolean
}

export class AudioPlayer {
  private context: AudioContext | null = null
  private nextPlayTime = 0
  private sessionQueue: AudioSession[] = []
  private receivingSession: AudioSession | null = null
  private activeSources = new Set<AudioBufferSourceNode>()
  private transitionSource: AudioBufferSourceNode | null = null

  onSpeakerChange?: (characterId: string | null, characterName: string | null) => void

  handleStreamStart(data: {
    character_id: string
    character_name: string
    message_id: string
    sample_rate: number
  }): void {
    this.ensureContext()

    const session: AudioSession = {
      characterId: data.character_id,
      characterName: data.character_name,
      messageId: data.message_id,
      sampleRate: data.sample_rate,
      pendingChunks: [],
      done: false,
    }

    const wasEmpty = this.sessionQueue.length === 0
    this.sessionQueue.push(session)
    this.receivingSession = session

    if (wasEmpty) {
      this.onSpeakerChange?.(session.characterId, session.characterName)
    }
  }

  handleAudioChunk(buffer: ArrayBuffer): void {
    if (!this.receivingSession) return

    if (this.sessionQueue[0] === this.receivingSession) {
      this.scheduleChunk(buffer, this.receivingSession.sampleRate)
    } else {
      this.receivingSession.pendingChunks.push(buffer)
    }
  }

  handleStreamStop(data: { character_id: string; message_id: string }): void {
    const stoppedSession = this.sessionQueue.find(
      (session) =>
        session.messageId === data.message_id &&
        session.characterId === data.character_id
    )
    if (!stoppedSession) return

    stoppedSession.done = true

    if (
      this.receivingSession &&
      this.receivingSession.messageId === data.message_id
    ) {
      this.receivingSession = null
    }

    if (this.sessionQueue[0] === stoppedSession) {
      this.scheduleTransition()
    }
  }

  interrupt(): void {
    this.stopAll()
  }

  stopAll(): void {
    for (const source of this.activeSources) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // source may have ended already
      }
      source.disconnect()
    }

    this.activeSources.clear()
    this.transitionSource = null
    this.sessionQueue = []
    this.receivingSession = null
    this.onSpeakerChange?.(null, null)

    if (this.context) {
      if (this.context.state === 'suspended') {
        void this.context.resume()
      }
      this.nextPlayTime = this.context.currentTime + 0.05
    } else {
      this.nextPlayTime = 0
    }
  }

  destroy(): void {
    this.stopAll()

    if (this.context) {
      void this.context.close()
      this.context = null
    }

    this.nextPlayTime = 0
  }

  private ensureContext(): void {
    if (this.context) {
      if (this.context.state === 'suspended') void this.context.resume()
      return
    }

    this.context = new AudioContext()
    this.nextPlayTime = this.context.currentTime + 0.05
  }

  private scheduleChunk(buffer: ArrayBuffer, sampleRate: number): void {
    const ctx = this.context
    if (!ctx) return

    const int16 = new Int16Array(buffer)
    if (int16.length === 0) return

    const float32 = new Float32Array(int16.length)
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768
    }

    const audioBuffer = ctx.createBuffer(1, float32.length, sampleRate)
    audioBuffer.copyToChannel(float32, 0)

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)
    this.registerSource(source)

    const startTime = Math.max(this.nextPlayTime, ctx.currentTime)
    source.start(startTime)
    this.nextPlayTime = startTime + audioBuffer.duration
  }

  private scheduleTransition(): void {
    const ctx = this.context
    if (!ctx) return
    if (this.transitionSource) return

    const sentinel = ctx.createBuffer(1, 1, ctx.sampleRate)
    const source = ctx.createBufferSource()
    source.buffer = sentinel
    source.connect(ctx.destination)
    this.transitionSource = source
    this.registerSource(source, () => this.advanceSession())
    source.start(Math.max(this.nextPlayTime, ctx.currentTime))
  }

  private advanceSession(): void {
    this.transitionSource = null
    this.sessionQueue.shift()

    if (this.sessionQueue.length === 0) {
      this.onSpeakerChange?.(null, null)
      return
    }

    const next = this.sessionQueue[0]
    this.onSpeakerChange?.(next.characterId, next.characterName)

    for (const chunk of next.pendingChunks) {
      this.scheduleChunk(chunk, next.sampleRate)
    }
    next.pendingChunks = []

    if (next.done) {
      this.scheduleTransition()
    }
  }

  private registerSource(
    source: AudioBufferSourceNode,
    onEnded?: () => void
  ): void {
    this.activeSources.add(source)
    source.onended = () => {
      this.activeSources.delete(source)
      if (this.transitionSource === source) {
        this.transitionSource = null
      }
      onEnded?.()
    }
  }
}
