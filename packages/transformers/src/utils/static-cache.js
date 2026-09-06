/**
 * Logical lengths of session-owned static KV tensors (their shapes describe capacity).
 * @type {WeakMap<object, {owner: {epoch: number, disposed: boolean}, epoch: number, length: number}>}
 */
const cacheInfo = new WeakMap();

/** @param {object} tensor @param {{epoch: number, disposed: boolean}} owner @param {number} length */
export function setStaticCacheInfo(tensor, owner, length) {
    cacheInfo.set(tensor, { owner, epoch: owner.epoch, length });
}

/** @param {import('onnxruntime-common').Tensor} tensor */
export function getStaticCacheInfo(tensor) {
    const info = cacheInfo.get(tensor);
    if (info && (info.owner.disposed || info.epoch !== info.owner.epoch || tensor.location === 'none')) {
        throw new Error('This static KV cache is no longer valid: its model was disposed or started a new sequence.');
    }
    return info;
}
