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

const makeAssets = (variant: DeepFilter3Variant): VariantAssets => {
    const suffix = variant === "webnn" ? ".webnn" : ""
    const tag = variant === "webnn" ? " (WebNN)" : ""
    return {
        encoder: {
            label: `DeepFilter3 encoder${tag}`,
            opfsPath: `${OPFS_DIR}/enc${suffix}.onnx`,
            urlPath: `${URL_DIR}/enc${suffix}.onnx`,
            approxBytes: 2_000_000
        },
        erbDecoder: {
            label: `DeepFilter3 ERB decoder${tag}`,
            opfsPath: `${OPFS_DIR}/erb_dec${suffix}.onnx`,
            urlPath: `${URL_DIR}/erb_dec${suffix}.onnx`,
            approxBytes: 3_300_000
        },
        dfDecoder: {
            label: `DeepFilter3 DF decoder${tag}`,
            opfsPath: `${OPFS_DIR}/df_dec${suffix}.onnx`,
            urlPath: `${URL_DIR}/df_dec${suffix}.onnx`,
            approxBytes: 3_300_000
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
