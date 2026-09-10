const isLocalhost = location.hostname === "localhost" || location.hostname === "127.0.0.1"

export const apiUrl = (path: string): string =>
    isLocalhost ? `/api-proxy${path}` : `https://api.opendaw.studio${path}`
