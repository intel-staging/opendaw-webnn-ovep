import {asDefined, Errors, Progress} from "@opendaw/lib-std"

export const fetchWithProgress = async (
    url: string,
    onProgress: Progress.Handler,
    signal: AbortSignal
): Promise<ArrayBuffer> => {
    const response = await fetch(url, {signal})
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching ${url}: ${response.statusText}`)
    }
    const contentLength = response.headers.get("content-length")
    const total = contentLength === null ? 0 : parseInt(contentLength, 10)
    const reader = asDefined(response.body, "No body in response").getReader()
    const chunks: Array<Uint8Array> = []
    let received = 0
    while (true) {
        if (signal.aborted) {
            await reader.cancel().catch(() => undefined)
            throw Errors.AbortError
        }
        const {done, value} = await reader.read()
        if (done) {break}
        chunks.push(value)
        received += value.length
        if (total > 0) {onProgress(received / total)}
    }
    const buffer = new Uint8Array(received)
    let offset = 0
    for (const chunk of chunks) {
        buffer.set(chunk, offset)
        offset += chunk.length
    }
    onProgress(1)
    return buffer.buffer
}
