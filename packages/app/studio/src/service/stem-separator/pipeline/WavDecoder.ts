import {TARGET_SAMPLE_RATE} from "./Constants"

export type DecodedStereo = {
    readonly left: Float32Array
    readonly right: Float32Array
    readonly sourceSampleRate: number
    readonly sourceNumberOfChannels: number
}

const decodeFileToAudioBuffer = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer()
    const ctx = new AudioContext()
    try {
        return await ctx.decodeAudioData(arrayBuffer)
    } finally {
        ctx.close().catch(() => undefined)
    }
}

const resampleTo44100 = async (audioBuffer: AudioBuffer): Promise<AudioBuffer> => {
    if (audioBuffer.sampleRate === TARGET_SAMPLE_RATE) {return audioBuffer}
    const targetLength = Math.ceil(audioBuffer.duration * TARGET_SAMPLE_RATE)
    const offline = new OfflineAudioContext(audioBuffer.numberOfChannels, targetLength, TARGET_SAMPLE_RATE)
    const source = offline.createBufferSource()
    source.buffer = audioBuffer
    source.connect(offline.destination)
    source.start(0)
    return await offline.startRendering()
}

const toStereo = (audioBuffer: AudioBuffer): {left: Float32Array, right: Float32Array} => {
    const channels = audioBuffer.numberOfChannels
    const length = audioBuffer.length
    if (channels === 1) {
        const mono = audioBuffer.getChannelData(0)
        const left = new Float32Array(length)
        const right = new Float32Array(length)
        left.set(mono)
        right.set(mono)
        return {left, right}
    }
    const leftSrc = audioBuffer.getChannelData(0)
    const rightSrc = audioBuffer.getChannelData(1)
    const left = new Float32Array(length)
    const right = new Float32Array(length)
    left.set(leftSrc)
    right.set(rightSrc)
    return {left, right}
}

export const decodeFileToStereo44100 = async (file: File): Promise<DecodedStereo> => {
    const decoded = await decodeFileToAudioBuffer(file)
    const sourceSampleRate = decoded.sampleRate
    const sourceNumberOfChannels = decoded.numberOfChannels
    const resampled = await resampleTo44100(decoded)
    const {left, right} = toStereo(resampled)
    return {left, right, sourceSampleRate, sourceNumberOfChannels}
}

export const resampleFramesToStereo44100 = async (
    frames: ReadonlyArray<Float32Array>,
    sampleRate: number
): Promise<{left: Float32Array, right: Float32Array}> => {
    const numChannels = Math.min(frames.length, 2)
    const length = frames[0].length
    const tempCtx = new AudioContext()
    let audioBuffer: AudioBuffer
    try {
        audioBuffer = tempCtx.createBuffer(numChannels, length, sampleRate)
        for (let ch = 0; ch < numChannels; ch++) {
            audioBuffer.copyToChannel(new Float32Array(frames[ch]), ch)
        }
    } finally {
        tempCtx.close().catch(() => undefined)
    }
    const resampled = await resampleTo44100(audioBuffer)
    return toStereo(resampled)
}
