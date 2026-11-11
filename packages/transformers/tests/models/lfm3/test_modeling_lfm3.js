import { PretrainedConfig } from "../../../src/transformers.js";
import { getCacheNames } from "../../../src/configs.js";
import { addPastKeyValues } from "../../../src/models/modeling_utils.js";

export default () => {
  describe("LFM3 caches", () => {
    const text_config = {
      model_type: "lfm3",
      layer_types: ["conv", "full_attention", "sliding_attention"],
      num_hidden_layers: 3,
      hidden_size: 16,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      head_dim: 4,
    };

    it.each([text_config, { model_type: "lfm3_vl", text_config }])("normalizes the text config and initializes both attention types for $model_type", (config) => {
      config = new PretrainedConfig(config);
      expect(config.normalized_config).toMatchObject({ num_layers: 3, num_heads: 2, dim_kv: 4, hidden_size: 16 });
      expect([...getCacheNames(config)]).toEqual(["past_conv.0", "past_key_values.1.key", "past_key_values.1.value", "past_key_values.2.key", "past_key_values.2.value"]);
      expect([...getCacheNames(config, { prefix: "present" })]).toEqual(["present_conv.0", "present.1.key", "present.1.value", "present.2.key", "present.2.value"]);

      const inputMetadata = [...getCacheNames(config)].map((name) => ({ name, type: "float16", shape: name.startsWith("past_conv") ? ["batch_size", 6, 16] : ["batch_size", 2, "past_sequence_length", 4] }));
      const feeds = { attention_mask: { dims: [2, 3] } };
      const cache = addPastKeyValues({ config, sessions: { model: { inputMetadata } } }, feeds, null);
      expect(feeds["past_conv.0"].dims).toEqual([2, 6, 16]);
      expect(feeds["past_key_values.1.key"].dims).toEqual([2, 2, 0, 4]);
      expect(feeds["past_key_values.2.value"].dims).toEqual([2, 2, 0, 4]);
      expect(feeds["past_conv.0"].type).toBe("float16");
      cache.dispose();
    });
  });
};
