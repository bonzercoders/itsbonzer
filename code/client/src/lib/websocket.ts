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

const PING_INTERVAL_MS = 15_000
const PONG_TIMEOUT_MS = 5_000
const RECONNECT_BASE_DELAY_MS = 1_000
const RECONNECT_MAX_DELAY_MS = 30_000
const MAX_RECONNECT_ATTEMPTS = 10

function resolveWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}

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

export class WebSocketClient {
  private socket: WebSocket | null = null
  private status: ConnectionStatus = 'disconnected'
  private connectPromise: Promise<void> | null = null
  private reconnectAttempts = 0
  private reconnectEnabled = true

  private pingIntervalId: number | null = null
  private pongTimeoutId: number | null = null
  private reconnectTimeoutId: number | null = null

  private readonly messageHandlers = new Set<MessageHandler>()
  private readonly binaryHandlers = new Set<BinaryHandler>()
  private readonly statusHandlers = new Set<StatusHandler>()
  private readonly errorHandlers = new Set<ErrorHandler>()

  connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    if (this.socket?.readyState === WebSocket.CONNECTING && this.connectPromise) {
      return this.connectPromise
    }

    this.reconnectEnabled = true

    this.connectPromise = new Promise<void>((resolve, reject) => {
      let didOpen = false
      let settled = false

      const resolveConnect = () => {
        if (settled) {
          return
        }

        settled = true
        resolve()
      }

      const rejectConnect = (error: unknown) => {
        if (settled) {
          return
        }

        settled = true
        reject(error)
      }

      const ws = new WebSocket(resolveWebSocketUrl())
      ws.binaryType = 'arraybuffer'

      this.socket = ws
      this.setStatus(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting')

      ws.onopen = () => {
        if (this.socket !== ws) {
          return
        }

        didOpen = true
        this.reconnectAttempts = 0
        this.stopReconnectTimer()
        this.setStatus('connected')
        this.startHeartbeat()
        resolveConnect()
      }

      ws.onclose = (event) => {
        if (this.socket === ws) {
          this.socket = null
        }

        this.stopHeartbeat()

        const shouldRetry = this.reconnectEnabled && event.code !== 1000
        if (shouldRetry) {
          this.scheduleReconnect()
        } else {
          this.stopReconnectTimer()
          this.setStatus('disconnected')
        }

        if (!didOpen) {
          rejectConnect(new Error('WebSocket connection closed before opening'))
        }
      }

      ws.onerror = () => {
        const error = new Error('WebSocket connection error')
        this.emitError(error)

        if (!didOpen) {
          rejectConnect(error)
        }
      }

      ws.onmessage = (event) => {
        void this.handleMessageEvent(event)
      }
    }).finally(() => {
      this.connectPromise = null
    })

    return this.connectPromise
  }

  disconnect(code = 1000, reason = 'Client disconnect'): void {
    this.reconnectEnabled = false
    this.reconnectAttempts = 0

    this.stopReconnectTimer()
    this.stopHeartbeat()

    if (this.socket) {
      this.socket.close(code, reason)
      this.socket = null
    }

    this.setStatus('disconnected')
  }

  send(message: OutboundMessage): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false
    }

    this.socket.send(JSON.stringify(message))
    return true
  }

  sendBinary(data: ArrayBuffer): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false
    }

    this.socket.send(data)
    return true
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
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
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

      if (!this.reconnectEnabled) {
        return
      }

      void this.connect().catch((error: unknown) => {
        this.emitError(error)
      })
    }, delay)
  }

  private startHeartbeat(): void {
    this.stopHeartbeat()

    this.pingIntervalId = window.setInterval(() => {
      const pingSent = this.send({ type: 'ping' })
      if (!pingSent) {
        return
      }

      if (this.pongTimeoutId !== null) {
        window.clearTimeout(this.pongTimeoutId)
      }

      this.pongTimeoutId = window.setTimeout(() => {
        this.pongTimeoutId = null

        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.close(4000, 'Pong timeout')
        }
      }, PONG_TIMEOUT_MS)
    }, PING_INTERVAL_MS)
  }

  private stopHeartbeat(): void {
    if (this.pingIntervalId !== null) {
      window.clearInterval(this.pingIntervalId)
      this.pingIntervalId = null
    }

    if (this.pongTimeoutId !== null) {
      window.clearTimeout(this.pongTimeoutId)
      this.pongTimeoutId = null
    }
  }

  private stopReconnectTimer(): void {
    if (this.reconnectTimeoutId !== null) {
      window.clearTimeout(this.reconnectTimeoutId)
      this.reconnectTimeoutId = null
    }
  }

  private handlePong(): void {
    if (this.pongTimeoutId !== null) {
      window.clearTimeout(this.pongTimeoutId)
      this.pongTimeoutId = null
    }
  }

  private async handleMessageEvent(event: MessageEvent): Promise<void> {
    if (event.data instanceof ArrayBuffer) {
      this.emitBinary(event.data)
      return
    }

    if (event.data instanceof Blob) {
      const buffer = await event.data.arrayBuffer()
      this.emitBinary(buffer)
      return
    }

    if (typeof event.data !== 'string') {
      this.emitError(new Error('Received unsupported WebSocket message type'))
      return
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(event.data)
    } catch (error) {
      this.emitError(error)
      return
    }

    if (!isInboundMessage(parsed)) {
      this.emitError(new Error('Received malformed WebSocket message'))
      return
    }

    if (parsed.type === 'pong') {
      this.handlePong()
      return
    }

    this.emitMessage(parsed)
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

  private emitMessage(message: InboundMessage): void {
    for (const handler of this.messageHandlers) {
      handler(message)
    }
  }

  private emitBinary(audio: ArrayBuffer): void {
    for (const handler of this.binaryHandlers) {
      handler(audio)
    }
  }

  private emitError(error: unknown): void {
    for (const handler of this.errorHandlers) {
      handler(error)
    }
  }
}
