export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'

export type ModelSettingsMessage = {
  type: 'model_settings'
  model: string
  temperature: number
  top_p: number
  min_p: number
  top_k: number
  frequency_penalty: number
  presence_penalty: number
  repetition_penalty: number
}

export type OutboundMessage =
  | { type: 'ping' }
  | { type: 'user_message'; text: string }
  | { type: 'start_listening' }
  | { type: 'stop_listening' }
  | { type: 'clear_history' }
  | { type: 'refresh_characters' }
  | ModelSettingsMessage

export type InboundMessage =
  | { type: 'pong' }
  | { type: 'stt_update'; text: string }
  | { type: 'stt_stabilized'; text: string }
  | { type: 'stt_final'; text: string }
  | {
      type: 'text_stream_start'
      data: { character_id: string; character_name: string; message_id: string }
    }
  | {
      type: 'text_chunk'
      data: {
        text: string
        character_id: string
        character_name: string
        message_id: string
      }
    }
  | {
      type: 'text_stream_stop'
      data: {
        character_id: string
        character_name: string
        message_id: string
        text: string
      }
    }
  | {
      type: 'audio_stream_start'
      data: {
        character_id: string
        character_name: string
        message_id: string
        sample_rate: number
      }
    }
  | {
      type: 'audio_stream_stop'
      data: { character_id: string; character_name: string; message_id: string }
    }

type MessageHandler = (message: InboundMessage) => void
type BinaryHandler = (audio: ArrayBuffer) => void
type StatusHandler = (status: ConnectionStatus) => void
type ErrorHandler = (error: unknown) => void

const ICE_SERVERS: RTCIceServer[] = [{ urls: 'stun:stun.l.google.com:19302' }]

const RECONNECT_BASE_DELAY_MS = 1_000
const RECONNECT_MAX_DELAY_MS = 30_000
const MAX_RECONNECT_ATTEMPTS = 10

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isTextMessageWithData(value: unknown): value is {
  type: 'text_stream_start' | 'text_chunk' | 'text_stream_stop'
  data: Record<string, unknown>
} {
  if (!isRecord(value) || typeof value.type !== 'string' || !isRecord(value.data)) {
    return false
  }

  return (
    value.type === 'text_stream_start' ||
    value.type === 'text_chunk' ||
    value.type === 'text_stream_stop'
  )
}

function isAudioMessageWithData(value: unknown): value is {
  type: 'audio_stream_start' | 'audio_stream_stop'
  data: Record<string, unknown>
} {
  if (!isRecord(value) || typeof value.type !== 'string' || !isRecord(value.data)) {
    return false
  }

  return value.type === 'audio_stream_start' || value.type === 'audio_stream_stop'
}

function isInboundMessage(value: unknown): value is InboundMessage {
  if (!isRecord(value) || typeof value.type !== 'string') {
    return false
  }

  if (value.type === 'pong') {
    return true
  }

  if (
    (value.type === 'stt_update' ||
      value.type === 'stt_stabilized' ||
      value.type === 'stt_final') &&
    typeof value.text === 'string'
  ) {
    return true
  }

  if (!isTextMessageWithData(value) && !isAudioMessageWithData(value)) {
    return false
  }

  if (
    (value.type === 'text_stream_start' || value.type === 'audio_stream_stop') &&
    typeof value.data.character_id === 'string' &&
    typeof value.data.character_name === 'string' &&
    typeof value.data.message_id === 'string'
  ) {
    return true
  }

  if (
    value.type === 'text_chunk' &&
    typeof value.data.text === 'string' &&
    typeof value.data.character_id === 'string' &&
    typeof value.data.character_name === 'string' &&
    typeof value.data.message_id === 'string'
  ) {
    return true
  }

  if (
    value.type === 'text_stream_stop' &&
    typeof value.data.text === 'string' &&
    typeof value.data.character_id === 'string' &&
    typeof value.data.character_name === 'string' &&
    typeof value.data.message_id === 'string'
  ) {
    return true
  }

  if (
    value.type === 'audio_stream_start' &&
    typeof value.data.character_id === 'string' &&
    typeof value.data.character_name === 'string' &&
    typeof value.data.message_id === 'string' &&
    typeof value.data.sample_rate === 'number'
  ) {
    return true
  }

  return false
}

export class WebRTCClient {
  private pc: RTCPeerConnection | null = null
  private dataChannel: RTCDataChannel | null = null
  private status: ConnectionStatus = 'disconnected'

  // Active mic track — preserved across reconnects so it can be re-attached
  private micTrack: MediaStreamTrack | null = null

  private reconnectAttempts = 0
  private reconnectEnabled = true
  private reconnectTimeoutId: number | null = null

  private readonly messageHandlers = new Set<MessageHandler>()
  private readonly binaryHandlers = new Set<BinaryHandler>()
  private readonly statusHandlers = new Set<StatusHandler>()
  private readonly errorHandlers = new Set<ErrorHandler>()

  async connect(): Promise<void> {
    if (this.status === 'connected' || this.status === 'connecting') {
      return
    }

    this.reconnectEnabled = true
    this.setStatus(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting')

    try {
      // Close any stale peer connection
      this.pc?.close()
      this.pc = new RTCPeerConnection({ iceServers: ICE_SERVERS })

      // Upstream audio transceiver (mic → server)
      const transceiver = this.pc.addTransceiver('audio', { direction: 'sendonly' })

      // If a live mic track exists from a previous session, re-attach it
      if (this.micTrack && this.micTrack.readyState === 'live') {
        await transceiver.sender.replaceTrack(this.micTrack)
      }

      // Downstream: DataChannel for TTS binary audio + JSON control messages
      this.dataChannel = this.pc.createDataChannel('control', { ordered: true })
      this.dataChannel.binaryType = 'arraybuffer'

      this.dataChannel.onopen = () => {
        this.reconnectAttempts = 0
        this.setStatus('connected')
      }

      this.dataChannel.onclose = () => {
        if (this.reconnectEnabled && this.status !== 'disconnected') {
          this.scheduleReconnect()
        }
      }

      this.dataChannel.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Binary: PCM16 TTS audio chunk
          for (const h of this.binaryHandlers) h(event.data)
        } else if (typeof event.data === 'string') {
          // Text: JSON control message
          let parsed: unknown
          try {
            parsed = JSON.parse(event.data)
          } catch {
            return
          }
          if (isInboundMessage(parsed)) {
            for (const h of this.messageHandlers) h(parsed)
          }
        }
      }

      this.pc.oniceconnectionstatechange = () => {
        if (this.pc?.iceConnectionState === 'failed') {
          this.scheduleReconnect()
        }
      }

      // Create offer, wait for ICE gathering to complete (vanilla ICE)
      const offer = await this.pc.createOffer()
      await this.pc.setLocalDescription(offer)
      await this.waitForIceGathering()

      // POST offer SDP to signaling endpoint
      const response = await fetch('/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: this.pc.localDescription!.sdp,
          type: this.pc.localDescription!.type,
        }),
      })

      if (!response.ok) {
        throw new Error(`Signaling failed: ${response.status}`)
      }

      const answer = (await response.json()) as { sdp: string; type: string }
      await this.pc.setRemoteDescription(new RTCSessionDescription(answer))
      // DataChannel.onopen fires once DTLS handshake completes → status → 'connected'
    } catch (error) {
      this.emitError(error)
      this.scheduleReconnect()
    }
  }

  disconnect(): void {
    this.reconnectEnabled = false
    this.reconnectAttempts = 0
    this.stopReconnectTimer()
    this.pc?.close()
    this.pc = null
    this.dataChannel = null
    this.setStatus('disconnected')
  }

  send(message: OutboundMessage): boolean {
    if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
      return false
    }

    this.dataChannel.send(JSON.stringify(message))
    return true
  }

  /**
   * Request microphone access, attach the track to the peer connection, and
   * return the live MediaStreamTrack for the caller to manage (mute/stop).
   */
  async enableMicrophone(): Promise<MediaStreamTrack> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 48000,
      },
    })

    const [track] = stream.getAudioTracks()
    this.micTrack = track

    const sender = this.pc?.getSenders().find(
      (s) => s.track?.kind === 'audio' || s.track === null
    )
    if (sender) {
      await sender.replaceTrack(track)
    }

    return track
  }

  /**
   * Stop the microphone track and detach it from the peer connection.
   */
  disableMicrophone(track: MediaStreamTrack): void {
    track.stop()

    if (this.micTrack === track) {
      this.micTrack = null
    }

    const sender = this.pc?.getSenders().find((s) => s.track === track)
    if (sender) {
      void sender.replaceTrack(null)
    }
  }

  getStatus(): ConnectionStatus {
    return this.status
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler)
    return () => {
      this.messageHandlers.delete(handler)
    }
  }

  onBinary(handler: BinaryHandler): () => void {
    this.binaryHandlers.add(handler)
    return () => {
      this.binaryHandlers.delete(handler)
    }
  }

  onStatusChange(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler)
    return () => {
      this.statusHandlers.delete(handler)
    }
  }

  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler)
    return () => {
      this.errorHandlers.delete(handler)
    }
  }

  private scheduleReconnect(): void {
    if (!this.reconnectEnabled || this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.setStatus('disconnected')
      return
    }

    this.stopReconnectTimer()
    this.setStatus('reconnecting')

    const delay = Math.min(
      RECONNECT_BASE_DELAY_MS * Math.pow(2, this.reconnectAttempts),
      RECONNECT_MAX_DELAY_MS
    )
    this.reconnectAttempts += 1

    this.reconnectTimeoutId = window.setTimeout(() => {
      this.reconnectTimeoutId = null
      if (this.reconnectEnabled) {
        void this.connect().catch((error: unknown) => this.emitError(error))
      }
    }, delay)
  }

  private stopReconnectTimer(): void {
    if (this.reconnectTimeoutId !== null) {
      window.clearTimeout(this.reconnectTimeoutId)
      this.reconnectTimeoutId = null
    }
  }

  private waitForIceGathering(): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.pc?.iceGatheringState === 'complete') {
        resolve()
        return
      }

      let resolved = false
      const tryResolve = () => {
        if (!resolved) {
          resolved = true
          resolve()
        }
      }

      this.pc!.addEventListener('icegatheringstatechange', () => {
        if (this.pc?.iceGatheringState === 'complete') tryResolve()
      })

      // Null candidate is the sentinel that ICE gathering is complete
      this.pc!.addEventListener('icecandidate', (e: RTCPeerConnectionIceEvent) => {
        if (e.candidate === null) tryResolve()
      })
    })
  }

  private setStatus(nextStatus: ConnectionStatus): void {
    if (this.status === nextStatus) {
      return
    }

    this.status = nextStatus
    for (const handler of this.statusHandlers) {
      handler(nextStatus)
    }
  }

  private emitError(error: unknown): void {
    for (const handler of this.errorHandlers) {
      handler(error)
    }
  }
}
