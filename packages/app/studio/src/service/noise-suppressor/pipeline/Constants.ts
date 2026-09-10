export const SAMPLE_RATE = 48000
export const FRAME_SIZE = 480
export const FFT_SIZE = 960
export const WINDOW_SIZE = 960
export const NUM_FREQS = 481
export const NB_ERB = 32
export const NB_DF = 96
export const LOOKAHEAD = 2
export const DF_ORDER = 5
export const DF_OFFSET = 2
export const ALPHA = 0.99
export const ATTEN_DENOM = 40.0

export const ERB_INDICES: ReadonlyArray<number> = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20,
    24, 28, 31, 37, 42, 50, 56, 67
]
