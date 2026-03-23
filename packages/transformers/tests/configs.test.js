import { AutoConfig, env } from "../src/transformers.js";
import { getFile } from "../src/utils/hub.js";

// Initialise the testing environment
env.allowLocalModels = false;
env.useFSCache = false;

const TEST_DATA = {
  "Xenova/bert-base-uncased": {
    model_type: "bert",
  },
};

describe("Configs", () => {
  for (const [model_id, minimal_config] of Object.entries(TEST_DATA)) {
    it(model_id, async () => {
      const config = await AutoConfig.from_pretrained(model_id);
      for (const [key, value] of Object.entries(minimal_config)) {
        expect(config[key]).toEqual(value);
      }
    });
  }
});
