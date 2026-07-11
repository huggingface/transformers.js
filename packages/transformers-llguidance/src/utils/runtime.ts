export function isNodeLikeRuntime() {
    return typeof process !== 'undefined' && Boolean(process.versions?.node);
}
