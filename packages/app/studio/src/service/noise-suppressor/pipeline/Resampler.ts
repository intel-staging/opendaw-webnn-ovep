import {SAMPLE_RATE} from "./Constants"

export const resampleTo48k = async (frames: ReadonlyArray<Float32Array>, sourceSampleRate: number): Promise<Float32Array[]> => {
    if (sourceSampleRate === SAMPLE_RATE) {
        return frames.map(ch => new Float32Array(ch))
    }
    const numChannels = frames.length
    const sourceLength = frames[0].length
    const targetLength = Math.ceil(sourceLength * SAMPLE_RATE / sourceSampleRate)
    const tempCtx = new AudioContext()
    let inputBuffer: AudioBuffer
    try {
        inputBuffer = tempCtx.createBuffer(numChannels, sourceLength, sourceSampleRate)
        for (let ch = 0; ch < numChannels; ch++) {
            inputBuffer.copyToChannel(new Float32Array(frames[ch]), ch)
        }
    } finally {
        tempCtx.close().catch(() => undefined)
    }
    const offline = new OfflineAudioContext(numChannels, targetLength, SAMPLE_RATE)
    const source = offline.createBufferSource()
    source.buffer = inputBuffer
    source.connect(offline.destination)
    source.start(0)
    const rendered = await offline.startRendering()
    const result: Float32Array[] = []
    for (let ch = 0; ch < numChannels; ch++) {
        const data = rendered.getChannelData(ch)
        result.push(new Float32Array(data))
    }
    return result
}

export const resampleFrom48k = async (frames: ReadonlyArray<Float32Array>, targetSampleRate: number): Promise<Float32Array[]> => {
    if (targetSampleRate === SAMPLE_RATE) {
        return frames.map(ch => new Float32Array(ch))
    }
    const numChannels = frames.length
    const sourceLength = frames[0].length
    const targetLength = Math.ceil(sourceLength * targetSampleRate / SAMPLE_RATE)
    const tempCtx = new AudioContext()
    let inputBuffer: AudioBuffer
    try {
        inputBuffer = tempCtx.createBuffer(numChannels, sourceLength, SAMPLE_RATE)
        for (let ch = 0; ch < numChannels; ch++) {
            inputBuffer.copyToChannel(new Float32Array(frames[ch]), ch)
        }
    } finally {
        tempCtx.close().catch(() => undefined)
    }
    const offline = new OfflineAudioContext(numChannels, targetLength, targetSampleRate)
    const source = offline.createBufferSource()
    source.buffer = inputBuffer
    source.connect(offline.destination)
    source.start(0)
    const rendered = await offline.startRendering()
    const result: Float32Array[] = []
    for (let ch = 0; ch < numChannels; ch++) {
        const data = rendered.getChannelData(ch)
        result.push(new Float32Array(data))
    }
    return result
}
