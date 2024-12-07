
// Adapted from https://www.npmjs.com/package/audiobuffer-to-wav
export function encodeWAV(samples, sampleRate = 16000) {
    let offset = 44;
    const buffer = new ArrayBuffer(offset + samples.length * 4);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF')
    /* RIFF chunk length */
    view.setUint32(4, 36 + samples.length * 4, true)
    /* RIFF type */
    writeString(view, 8, 'WAVE')
    /* format chunk identifier */
    writeString(view, 12, 'fmt ')
    /* format chunk length */
    view.setUint32(16, 16, true)
    /* sample format (raw) */
    view.setUint16(20, 3, true)
    /* channel count */
    view.setUint16(22, 1, true)
    /* sample rate */
    view.setUint32(24, sampleRate, true)
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 4, true)
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 4, true)
    /* bits per sample */
    view.setUint16(34, 32, true)
    /* data chunk identifier */
    writeString(view, 36, 'data')
    /* data chunk length */
    view.setUint32(40, samples.length * 4, true)

    for (let i = 0; i < samples.length; ++i, offset += 4) {
        view.setFloat32(offset, samples[i], true)
    }

    return buffer
}
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; ++i) {
        view.setUint8(offset + i, string.charCodeAt(i))
    }
}

export async function share(body, settings) {
    const response = await fetch('https://huggingface.co/uploads', { method: 'POST', body });
    if (!response.ok) throw new Error(`Failed to upload audio: ${response.statusText}`);
    const url = await response.text();

    const params = new URLSearchParams({
        title: `🎵 ${settings.prompt}`,
        description: `<audio controls src="${url}"></audio>\n${JSON.stringify(settings, null, 2)}`,
    });

    const shareURL = `https://huggingface.co/spaces/Xenova/musicgen-web/discussions/new?${params.toString()}`;
    window.open(shareURL, '_blank');
}