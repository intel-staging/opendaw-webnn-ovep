import {ERB_INDICES, NB_ERB, NUM_FREQS} from "./Constants"

export const computeErb = (mag: Float32Array): Float32Array => {
    const erb = new Float32Array(NB_ERB)
    let binOffset = 0
    for (let band = 0; band < NB_ERB; band++) {
        const width = ERB_INDICES[band]
        let sum = 0
        for (let k = 0; k < width; k++) {
            sum += mag[binOffset + k]
        }
        erb[band] = sum / width
        binOffset += width
    }
    return erb
}

export const expandMask = (erbMask: Float32Array): Float32Array => {
    const mask = new Float32Array(NUM_FREQS)
    let binOffset = 0
    for (let band = 0; band < NB_ERB; band++) {
        const width = ERB_INDICES[band]
        const val = erbMask[band]
        for (let k = 0; k < width; k++) {
            mask[binOffset + k] = val
        }
        binOffset += width
    }
    return mask
}
