/** @private Shared capability check for URL audio loading and byte decoding. */
export function assertAudioContextAvailable() {
    if (typeof AudioContext === 'undefined') {
        throw Error(
            'Unable to load audio from path/URL since `AudioContext` is not available in your environment. ' +
                'Instead, audio data should be passed directly to the pipeline/processor. ' +
                'For more information and some example code, see https://huggingface.co/docs/transformers.js/guides/node-audio-processing.',
        );
    }
}
