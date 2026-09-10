const username = "openDAW"
const password = "prototype"
export const base64Credentials = btoa(`${username}:${password}`)
export const OpenDAWHeaders: RequestInit = {
    method: "GET",
    headers: {"Authorization": `Basic ${base64Credentials}`}
}

const isLocalhost = typeof location !== "undefined"
    && (location.hostname === "localhost" || location.hostname === "127.0.0.1")

export const openDAWApiRoot = (path: string): string =>
    isLocalhost ? `/api-proxy${path}` : `https://api.opendaw.studio${path}`