import {ALPHA, ATTEN_DENOM, NB_ERB, NUM_FREQS} from "./Constants"

export type NormalizerState = {
    erbState: Float32Array
    unitState: Float32Array
}

const MEAN_NORM_INIT = -60.0
const UNIT_NORM_INIT = 0.001

export const createNormalizerState = (): NormalizerState => ({
    erbState: new Float32Array(NB_ERB).fill(MEAN_NORM_INIT),
    unitState: new Float32Array(NUM_FREQS).fill(UNIT_NORM_INIT)
})

export const normalizeErb = (erbLinear: Float32Array, state: NormalizerState): Float32Array => {
    const out = new Float32Array(NB_ERB)
    for (let i = 0; i < NB_ERB; i++) {
        const db = 10 * Math.log10(erbLinear[i] + 1e-10)
        state.erbState[i] = ALPHA * state.erbState[i] + (1 - ALPHA) * db
        out[i] = (db - state.erbState[i]) / ATTEN_DENOM
    }
    return out
}

export const normalizeSpec = (
    specReal: Float32Array,
    specImag: Float32Array,
    state: NormalizerState
): Float32Array => {
    const out = new Float32Array(NUM_FREQS * 2)
    for (let i = 0; i < NUM_FREQS; i++) {
        const re = specReal[i]
        const im = specImag[i]
        const mag = Math.sqrt(re * re + im * im)
        state.unitState[i] = ALPHA * state.unitState[i] + (1 - ALPHA) * mag
        const denom = Math.sqrt(state.unitState[i]) + 1e-10
        out[i * 2] = re / denom
        out[i * 2 + 1] = im / denom
    }
    return out
}
