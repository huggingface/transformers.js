// Independent pre-beam implementation with PR #1809's two indexing changes.
// https://github.com/huggingface/transformers.js/pull/1809
import { Tensor, cat, stack, std_mean, mean } from "../../../src/utils/tensor.js";
import { medianFilter, dynamic_time_warping } from "../../../src/utils/maths.js";
import { mergeArrays } from "../../../src/utils/core.js";
import { logger } from "../../../src/utils/logger.js";
export function upstream1809(generate_outputs, alignment_heads, num_frames = null, time_precision = 0.02, num_input_ids = 0) {
  if (!generate_outputs.cross_attentions) {
    throw new Error("Model outputs must contain cross attentions to extract timestamps. " + "This is most likely because the model was not exported with `output_attentions=True`.");
  }
  if (num_frames == null) {
    logger.warn("`num_frames` has not been set, meaning the entire audio will be analyzed. " + "This may lead to inaccurate token-level timestamps for short audios (< 30 seconds).");
  }

  // @ts-expect-error TS2339
  let median_filter_width = this.config.median_filter_width;
  if (median_filter_width === undefined) {
    logger.warn("Model config has no `median_filter_width`, using default value of 7.");
    median_filter_width = 7;
  }

  // TODO: Improve batch processing
  const batch = generate_outputs.cross_attentions;
  // Create a list with `decoder_layers` elements, each a tensor of shape
  // (batch size, attention_heads, output length, input length).
  const cross_attentions = Array.from(
    // @ts-expect-error TS2339
    { length: this.config.decoder_layers },
    // Concatenate the cross attentions for each layer across sequence length dimension.
    (_, i) =>
      cat(
        batch.map((x) => x[i]),
        2,
      ),
  );

  const weights = stack(
    alignment_heads.map(([l, h]) => {
      if (l >= cross_attentions.length) {
        throw new Error(`Layer index ${l} is out of bounds for cross attentions (length ${cross_attentions.length}).`);
      }
      return num_frames ? cross_attentions[l].slice(null, h, null, [0, num_frames]) : cross_attentions[l].slice(null, h);
    }),
  ).permute(1, 0, 2, 3);

  const [std, calculatedMean] = std_mean(weights, -2, 0, true);

  // Normalize and smoothen the weights.
  const smoothedWeights = weights.clone(); // [1, 8, seqLength, 1500]

  for (let a = 0; a < smoothedWeights.dims[0]; ++a) {
    const aTensor = smoothedWeights[a]; // [8, seqLength, 1500]

    for (let b = 0; b < aTensor.dims[0]; ++b) {
      const bTensor = aTensor[b]; // [seqLength, 1500]

      const stdTensorData = std[a][b][0].data; // [1500]
      const meanTensorData = calculatedMean[a][b][0].data; // [1500]

      for (let c = 0; c < bTensor.dims[0]; ++c) {
        let cTensorData = bTensor[c].data; // [1500]
        for (let d = 0; d < cTensorData.length; ++d) {
          cTensorData[d] = (cTensorData[d] - meanTensorData[d]) / stdTensorData[d];
        }

        // Apply median filter.
        cTensorData.set(medianFilter(cTensorData, median_filter_width));
      }
    }
  }

  // Keep the final prompt query, which predicts the first generated token.
  const croppedWeights = num_input_ids > 1 ? smoothedWeights.slice(null, null, [num_input_ids - 1, smoothedWeights.dims[2]], null) : smoothedWeights;

  // Average the different cross-attention heads.
  const batchedMatrices = [mean(croppedWeights, 1)];

  const timestampsShape = generate_outputs.sequences.dims;

  const timestamps = new Tensor("float32", new Float32Array(timestampsShape[0] * timestampsShape[1]), timestampsShape);

  // Perform dynamic time warping on each element of the batch.
  for (let batch_idx = 0; batch_idx < timestampsShape[0]; ++batch_idx) {
    // NOTE: Since we run only one batch at a time, we can squeeze to get the same dimensions
    // as the python implementation
    const matrix = batchedMatrices[batch_idx].neg().squeeze_(0);
    const [text_indices, time_indices] = dynamic_time_warping(matrix.tolist());

    const diffs = Array.from({ length: text_indices.length - 1 }, (v, i) => text_indices[i + 1] - text_indices[i]);
    const jumps = mergeArrays([1], diffs).map((x) => !!x); // convert to boolean

    const jump_times = [];
    for (let i = 0; i < jumps.length; ++i) {
      if (jumps[i]) {
        // NOTE: No point in rounding here, since we set to Float32Array later
        jump_times.push(time_indices[i] * time_precision);
      }
    }

    // Prompt tokens get zero; each generated token gets its predicting row time.
    const padded = new Array(Math.max(num_input_ids - 1, 0) + 1).fill(0);
    padded.push(...jump_times);
    timestamps[batch_idx].data.set(padded.slice(0, timestampsShape[1]));
  }

  return timestamps;
}
