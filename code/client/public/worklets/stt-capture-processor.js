class STTCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const processorOptions = options?.processorOptions ?? {};
    this.inputSampleRate = sampleRate;
    this.targetSampleRate = processorOptions.targetSampleRate ?? 16000;
    this.frameDurationMs = processorOptions.frameDurationMs ?? 20;

    this.frameSamples = Math.max(
      1,
      Math.round((this.targetSampleRate * this.frameDurationMs) / 1000)
    );
    this.resampleStep = this.inputSampleRate / this.targetSampleRate;
    this.resampleCursor = 0;

    this.inputBuffer = [];
    this.resampledBuffer = [];
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0 || channels[0].length === 0) {
      return true;
    }

    const channelCount = channels.length;
    const frameLength = channels[0].length;

    for (let i = 0; i < frameLength; i += 1) {
      let sum = 0;
      for (let c = 0; c < channelCount; c += 1) {
        sum += channels[c][i] || 0;
      }
      this.inputBuffer.push(sum / channelCount);
    }

    this.resampleInput();
    this.emitFrames();
    return true;
  }

  resampleInput() {
    while (this.resampleCursor + this.resampleStep <= this.inputBuffer.length - 1) {
      const baseIndex = Math.floor(this.resampleCursor);
      const fraction = this.resampleCursor - baseIndex;
      const sampleA = this.inputBuffer[baseIndex];
      const sampleB = this.inputBuffer[baseIndex + 1] ?? sampleA;
      const interpolated = sampleA + (sampleB - sampleA) * fraction;
      this.resampledBuffer.push(interpolated);
      this.resampleCursor += this.resampleStep;
    }

    const consumed = Math.floor(this.resampleCursor);
    if (consumed > 0) {
      this.inputBuffer = this.inputBuffer.slice(consumed);
      this.resampleCursor -= consumed;
    }
  }

  emitFrames() {
    while (this.resampledBuffer.length >= this.frameSamples) {
      const frame = this.resampledBuffer.slice(0, this.frameSamples);
      this.resampledBuffer = this.resampledBuffer.slice(this.frameSamples);

      const pcm16 = new Int16Array(frame.length);
      for (let i = 0; i < frame.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, frame[i]));
        pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
      }

      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
  }
}

registerProcessor("stt-capture-processor", STTCaptureProcessor);
