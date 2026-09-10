import {BluesteinFft} from "./BluesteinFft"
import {FRAME_SIZE, FFT_SIZE, NUM_FREQS} from "./Constants"
import {getVorbisWindow} from "./VorbisWindow"

export type SynthesisState = {
    mem: Float32Array
    fft: BluesteinFft
}

export const createSynthesisState = (fft: BluesteinFft): SynthesisState => ({
    mem: new Float32Array(FRAME_SIZE),
    fft
})

export const frameSynthesis = (
    specReal: Float32Array,
    specImag: Float32Array,
    state: SynthesisState
): Float32Array => {
    const win = getVorbisWindow()
    const fullReal = new Float32Array(FFT_SIZE)
    const fullImag = new Float32Array(FFT_SIZE)
    fullReal.set(specReal)
    fullImag.set(specImag)
    for (let i = 1; i < NUM_FREQS - 1; i++) {
        fullReal[FFT_SIZE - i] = specReal[i]
        fullImag[FFT_SIZE - i] = -specImag[i]
    }
    const timeFrame = state.fft.inverse(fullReal, fullImag)
    for (let i = 0; i < FFT_SIZE; i++) {
        timeFrame[i] *= win[i] * FFT_SIZE
    }
    const out = new Float32Array(FRAME_SIZE)
    for (let i = 0; i < FRAME_SIZE; i++) {
        out[i] = timeFrame[i] + state.mem[i]
    }
    state.mem.set(timeFrame.subarray(FRAME_SIZE))
    return out
}
