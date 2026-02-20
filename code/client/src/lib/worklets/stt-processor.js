class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()

    const processorOptions = options.processorOptions || {}
    this.targetSampleRate = processorOptions.targetSampleRate || 16000
    this.sourceSampleRate = sampleRate
    this.step = this.sourceSampleRate / this.targetSampleRate

    this.readIndex = 0
    this.inputCarry = new Float32Array(0)
    this.outputBuffer = []
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || !input[0] || input[0].length === 0) {
      return true
    }

    const channelData = input[0]
    const merged = new Float32Array(this.inputCarry.length + channelData.length)
    merged.set(this.inputCarry, 0)
    merged.set(channelData, this.inputCarry.length)

    let position = this.readIndex
    while (position + 1 < merged.length) {
      const index = Math.floor(position)
      const fraction = position - index
      const sample =
        merged[index] + (merged[index + 1] - merged[index]) * fraction

      const clamped = Math.max(-1, Math.min(1, sample))
      this.outputBuffer.push(Math.round(clamped * 32767))
      position += this.step
    }

    const consumed = Math.floor(position)
    this.readIndex = position - consumed
    this.inputCarry = merged.slice(consumed)

    while (this.outputBuffer.length >= 320) {
      const chunk = new Int16Array(this.outputBuffer.splice(0, 320))
      this.port.postMessage(chunk.buffer, [chunk.buffer])
    }

    return true
  }
}

registerProcessor('stt-processor', STTProcessor)
