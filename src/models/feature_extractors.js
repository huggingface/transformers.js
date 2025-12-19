export * from './model-processors/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.js';
export * from './model-processors/encodec/feature_extraction_encodec.js';
export * from './model-processors/chatterbox/feature_extraction_chatterbox.js';
export * from './model-processors/clap/feature_extraction_clap.js';
export * from './model-processors/dac/feature_extraction_dac.js';
export * from './model-processors/gemma3n/feature_extraction_gemma3n.js';
export * from './model-processors/moonshine/feature_extraction_moonshine.js';
export * from './model-processors/parakeet/feature_extraction_parakeet.js';
export * from './model-processors/pyannote/feature_extraction_pyannote.js';
export * from './model-processors/seamless_m4t/feature_extraction_seamless_m4t.js';
export * from './model-processors/snac/feature_extraction_snac.js';
export * from './model-processors/speecht5/feature_extraction_speecht5.js';
export * from './model-processors/wav2vec2/feature_extraction_wav2vec2.js';
export * from './model-processors/wespeaker/feature_extraction_wespeaker.js';
export * from './model-processors/whisper/feature_extraction_whisper.js';

// For legacy support, ImageFeatureExtractor is an alias for ImageProcessor
export { ImageProcessor as ImageFeatureExtractor } from '../base/image_processors_utils.js';
