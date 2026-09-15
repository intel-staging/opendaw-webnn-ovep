import {ModelAsset} from "../stem-separator/ModelAssets"
export type {ModelAsset}

export type DeepFilter3Variant = "original" | "webnn"

const OPFS_DIR = "models/deepfilter/df3"
const URL_DIR = "models/df3"

type VariantAssets = {
    readonly encoder: ModelAsset
    readonly erbDecoder: ModelAsset
    readonly dfDecoder: ModelAsset
}

// Exact sizes reject a truncated OPFS entry. Only for the upstream originals, which are
// byte-stable. The webnn variants are generated locally by scripts/rewrite_gru_for_webnn.py
// and their size depends on the onnx library version, so pinning them would cause false
// rejections. Left undefined = no validation.
const EXACT_BYTES: Record<DeepFilter3Variant, Partial<Record<keyof VariantAssets, number>>> = {
    original: {encoder: 1_954_042, erbDecoder: 3_292_397, dfDecoder: 3_340_803},
    webnn: {}
}

const makeAssets = (variant: DeepFilter3Variant): VariantAssets => {
    const suffix = variant === "webnn" ? ".webnn" : ""
    const tag = variant === "webnn" ? " (WebNN)" : ""
    const exact = EXACT_BYTES[variant]
    return {
        encoder: {
            label: `DeepFilter3 encoder${tag}`,
            opfsPath: `${OPFS_DIR}/enc${suffix}.onnx`,
            urlPath: `${URL_DIR}/enc${suffix}.onnx`,
            approxBytes: 2_000_000,
            exactBytes: exact.encoder
        },
        erbDecoder: {
            label: `DeepFilter3 ERB decoder${tag}`,
            opfsPath: `${OPFS_DIR}/erb_dec${suffix}.onnx`,
            urlPath: `${URL_DIR}/erb_dec${suffix}.onnx`,
            approxBytes: 3_300_000,
            exactBytes: exact.erbDecoder
        },
        dfDecoder: {
            label: `DeepFilter3 DF decoder${tag}`,
            opfsPath: `${OPFS_DIR}/df_dec${suffix}.onnx`,
            urlPath: `${URL_DIR}/df_dec${suffix}.onnx`,
            approxBytes: 3_300_000,
            exactBytes: exact.dfDecoder
        }
    }
}

const OriginalAssets = makeAssets("original")
const WebnnAssets = makeAssets("webnn")

export const DeepFilter3Encoder = OriginalAssets.encoder
export const DeepFilter3ErbDecoder = OriginalAssets.erbDecoder
export const DeepFilter3DfDecoder = OriginalAssets.dfDecoder

export const DeepFilter3Assets = [DeepFilter3Encoder, DeepFilter3ErbDecoder, DeepFilter3DfDecoder]

export const getDeepFilter3Assets = (variant: DeepFilter3Variant): VariantAssets =>
    variant === "webnn" ? WebnnAssets : OriginalAssets

export const allDeepFilter3Assets: ReadonlyArray<ModelAsset> = [
    OriginalAssets.encoder, OriginalAssets.erbDecoder, OriginalAssets.dfDecoder,
    WebnnAssets.encoder, WebnnAssets.erbDecoder, WebnnAssets.dfDecoder
]
