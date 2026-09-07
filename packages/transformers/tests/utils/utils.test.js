import { AutoProcessor } from "../../src/transformers.js";
import { hamming, hanning, mel_filter_bank } from "../../src/utils/audio.js";
import { getFile } from "../../src/utils/hub.js";
import { toAbsoluteURL } from "../../src/utils/hub/utils.js";
import { RawImage } from "../../src/utils/image.js";

import { load_cached_image } from "../asset_cache.js";
import { MAX_TEST_EXECUTION_TIME } from "../init.js";

describe("Utilities", () => {
  describe("Audio utilities", () => {
    it(
      "should calculate MEL filters",
      async () => {
        // NOTE: Uses official HF implementation as reference:
        const processor = await AutoProcessor.from_pretrained("openai/whisper-tiny.en");
        const config = processor.feature_extractor.config;

        // True MEL filters
        const original_mel_filters = config.mel_filters;

        // Calculated MEL filters
        const calculated_mel_filters = mel_filter_bank(
          Math.floor(1 + config.n_fft / 2), // num_frequency_bins
          config.feature_size, // num_mel_filters
          0.0, // min_frequency
          8000.0, // max_frequency
          config.sampling_rate, // sampling_rate
          "slaney", // norm
          "slaney", // mel_scale
        );

        const original = original_mel_filters.flat();
        const calculated = calculated_mel_filters.flat();

        // Compute max difference
        const maxdiff = original.reduce((maxdiff, _, i) => {
          const diff = Math.abs(original[i] - calculated[i]);
          return Math.max(maxdiff, diff);
        }, -Infinity);
        expect(maxdiff).toBeGreaterThanOrEqual(0);
        expect(maxdiff).toBeLessThan(1e-6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should calculate window",
      async () => {
        expect(hanning(10)).toBeCloseToNested(new Float64Array([0.0, 0.11697777844051105, 0.41317591116653485, 0.75, 0.9698463103929542, 0.9698463103929542, 0.75, 0.41317591116653485, 0.11697777844051105, 0.0]));
        expect(hamming(10)).toBeCloseToNested(new Float64Array([0.08000000000000002, 0.1876195561652702, 0.46012183827321207, 0.7700000000000001, 0.9722586055615179, 0.9722586055615179, 0.7700000000000001, 0.46012183827321207, 0.1876195561652702, 0.08000000000000002]));
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Hub utilities", () => {
    it("Read data from blob", async () => {
      const blob = new Blob(["Hello, world!"], { type: "text/plain" });
      const blobUrl = URL.createObjectURL(blob);
      const data = await getFile(blobUrl);
      expect(await data.text()).toBe("Hello, world!");
    });

    describe("toAbsoluteURL", () => {
      const originalLocation = globalThis.location;

      afterEach(() => {
        globalThis.location = originalLocation;
      });

      it("resolves a relative URL against the page", () => {
        globalThis.location = { href: "https://example.com/app/index.html" };
        expect(toAbsoluteURL("models/")).toBe("https://example.com/app/models/");
      });

      it("throws for a malformed URL, so a bad env.backends.onnx.wasm.wasmPaths is not passed on", () => {
        globalThis.location = { href: "https://example.com/app/index.html" };
        expect(() => toAbsoluteURL("https://bad host/wasm/")).toThrow(/Invalid URL/);
      });

      it("throws for a relative URL on an opaque-origin page by default", () => {
        globalThis.location = { href: "about:blank" };
        expect(() => toAbsoluteURL("models/")).toThrow(/Invalid URL/);
      });

      it("returns the URL unchanged instead of throwing when allowUnresolved is set", () => {
        globalThis.location = { href: "about:blank" };
        expect(toAbsoluteURL("models/", { allowUnresolved: true })).toBe("models/");
        // The caller checks the result with isValidUrl, which rejects anything unresolved
        expect(toAbsoluteURL("https://bad host/wasm/", { allowUnresolved: true })).toBe("https://bad host/wasm/");
      });
    });
  });

  describe("Image utilities", () => {
    const [width, height, channels] = [2, 2, 3];
    const data = Uint8Array.from({ length: width * height * channels }, (_, i) => i % 5);
    const tiny_image = new RawImage(data, width, height, channels);

    let image;
    beforeAll(async () => {
      image = await load_cached_image("checkerboard_64x32");
    });

    it("Can split image into separate channels", async () => {
      const image_data = tiny_image.split().map((x) => x.data);

      const target = [
        new Uint8Array([0, 3, 1, 4]), // Reds
        new Uint8Array([1, 4, 2, 0]), // Greens
        new Uint8Array([2, 0, 3, 1]), // Blues
      ];

      expect(image_data).toEqual(target);
    });

    it("Can splits channels for grayscale", async () => {
      const image_data = tiny_image
        .grayscale()
        .split()
        .map((x) => x.data);
      const target = [new Uint8ClampedArray([1, 3, 2, 1])];

      expect(image_data).toEqual(target);
    });

    it("Read image from URL", async () => {
      expect(image.width).toBe(64);
      expect(image.height).toBe(32);
      expect(image.channels).toBe(3);
    });

    it("Can resize image", async () => {
      const resized = await image.resize(32, 16);
      expect(resized.width).toBe(32);
      expect(resized.height).toBe(16);
    });

    it("Can resize with aspect ratio", async () => {
      const resized = await image.resize(32, null);
      expect(resized.width).toBe(32);
      expect(resized.height).toBe(16);
    });

    it("Returns original image if width and height are null", async () => {
      const resized = await image.resize(null, null);
      expect(resized.width).toBe(64);
      expect(resized.height).toBe(32);
    });
  });
});
