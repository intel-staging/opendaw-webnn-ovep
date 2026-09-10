export type ModelAsset = {
    readonly label: string
    readonly opfsPath: string
    readonly urlPath: string
    readonly approxBytes: number
}

const MODEL_DIR = "models/htdemucs_fwd"

export const HtdemucsFwdGraph: ModelAsset = {
    label: "htdemucs_fwd graph",
    opfsPath: `${MODEL_DIR}/htdemucs_fwd.onnx`,
    urlPath: `models/${MODEL_DIR}/htdemucs_fwd.onnx`,
    approxBytes: 2_400_000
}

export const HtdemucsFwdWeights: ModelAsset = {
    label: "htdemucs_fwd weights",
    opfsPath: `${MODEL_DIR}/htdemucs_fwd.onnx.data`,
    urlPath: `models/${MODEL_DIR}/htdemucs_fwd.onnx.data`,
    approxBytes: 176_000_000
}

export const HtdemucsFwdAssets: ReadonlyArray<ModelAsset> = [HtdemucsFwdGraph, HtdemucsFwdWeights]

export const ModelAssetWeights = HtdemucsFwdAssets.map(asset => asset.approxBytes)
