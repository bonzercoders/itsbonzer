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
      // First session in queue — it is immediately the active speaker.
      this.onSpeakerChange?.(session.characterId, session.characterName)
    }
    // If the queue was not empty, this session is queued behind the playing one.
    // onSpeakerChange fires later from advanceSession().
  }

  handleAudioChunk(buffer: ArrayBuffer): void {
    if (!this.receivingSession) return

    if (this.sessionQueue[0] === this.receivingSession) {
      // Active session: schedule immediately for low-latency playback.
      this.scheduleChunk(buffer, this.receivingSession.sampleRate)
    } else {
      // Queued session: buffer until this session becomes active.
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
      // The session that just finished receiving IS the currently playing one.
      // All its chunks have already been scheduled — schedule the transition sentinel.
      this.scheduleTransition()
    }
    // Else: stoppedSession is still queued behind another playing session.
    // When advanceSession() eventually promotes it, it will see done=true and
    // schedule the sentinel immediately.
  }

  destroy(): void {
    this.sessionQueue = []
    this.receivingSession = null
    this.nextPlayTime = 0
    if (this.context) {
      void this.context.close()
      this.context = null
    }
  }

  private ensureContext(): void {
    if (this.context) {
      // AudioContext can be auto-suspended when the tab is backgrounded.
      if (this.context.state === 'suspended') void this.context.resume()
      return
    }
    this.context = new AudioContext()
    // 50 ms of runway so the first chunk doesn't miss its start time.
    this.nextPlayTime = this.context.currentTime + 0.05
  }

  private scheduleChunk(buffer: ArrayBuffer, sampleRate: number): void {
    const ctx = this.context
    if (!ctx) return

    // PCM16: signed 16-bit integers, 1 channel.
    const int16 = new Int16Array(buffer)
    if (int16.length === 0) return

    // Web Audio API requires Float32 in [-1, 1].
    const float32 = new Float32Array(int16.length)
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768
    }

    const audioBuffer = ctx.createBuffer(1, float32.length, sampleRate)
    audioBuffer.copyToChannel(float32, 0)

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)

    // Clamp: if the tab was backgrounded, ctx.currentTime may have advanced past
    // nextPlayTime. Snap the write head forward to avoid scheduling in the past.
    const startTime = Math.max(this.nextPlayTime, ctx.currentTime)
    source.start(startTime)
    this.nextPlayTime = startTime + audioBuffer.duration
  }

  private scheduleTransition(): void {
    const ctx = this.context
    if (!ctx) return

    // A 1-sample silent buffer scheduled at the write head. Its onended fires
    // after all real audio for this session has finished playing.
    const sentinel = ctx.createBuffer(1, 1, ctx.sampleRate)
    const source = ctx.createBufferSource()
    source.buffer = sentinel
    source.connect(ctx.destination)
    source.start(Math.max(this.nextPlayTime, ctx.currentTime))
    source.onended = () => this.advanceSession()
  }

  private advanceSession(): void {
    this.sessionQueue.shift()

    if (this.sessionQueue.length === 0) {
      this.onSpeakerChange?.(null, null)
      return
    }

    const next = this.sessionQueue[0]
    this.onSpeakerChange?.(next.characterId, next.characterName)

    // Drain chunks that arrived while this session was queued.
    for (const chunk of next.pendingChunks) {
      this.scheduleChunk(chunk, next.sampleRate)
    }
    next.pendingChunks = []

    if (next.done) {
      // audio_stream_stop already arrived for this session — schedule sentinel now.
      this.scheduleTransition()
    }
    // Else: handleStreamStop will be called later. It will see sessionQueue[0] === next
    // and schedule the sentinel then.
  }
}
