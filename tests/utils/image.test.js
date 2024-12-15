import { RawImage, rand } from "../../src/transformers.js";
import { load_cached_image } from "../asset_cache.js";

const TEST_IMAGES = {
  rgba: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), 2, 3, 4),
  rgb: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 2, 3, 3),
  la: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 2, 3, 2),
  l: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5]), 2, 3, 1),
};

describe("Image utilities", () => {
  describe("Padding", () => {
    it("should pad image", async () => {
      /** @type {RawImage} */
      const padded_image = await load_cached_image("blue_image")
        .then((image) => image.resize(224, 224))
        .then((image) => image.pad([128, 128, 128, 128]));

      expect(padded_image.size).toEqual([480, 480]);

      const avg = padded_image.data.reduce((acc, val) => acc + val, 0) / padded_image.data.length;
      expect(avg).toBeCloseTo((224 * 224 * 255) / (3 * 480 * 480), 6);
    });
  });

  describe("Tensor to Image", () => {
    it("should create an image from a tensor (CHW)", () => {
      const tensor_chw = rand([3, 128, 256]).mul_(255).to("uint8");
      const image = RawImage.fromTensor(tensor_chw);
      expect(image.size).toEqual([256, 128]);
    });
    it("should create an image from a tensor (HWC)", () => {
      const tensor_hwc = rand([128, 256, 3]).mul_(255).to("uint8");
      const image = RawImage.fromTensor(tensor_hwc, "HWC");
      expect(image.size).toEqual([256, 128]);
    });
  });

  describe("Channel conversions", () => {
    it("should convert RGBA to L (grayscale)", async () => {
      const grayscale = TEST_IMAGES.rgba.clone().grayscale();
      expect(grayscale.size).toEqual(TEST_IMAGES.rgba.size);
      expect(grayscale.channels).toEqual(1);
    });

    it("should convert RGB to L (grayscale)", async () => {
      const grayscale = TEST_IMAGES.rgb.clone().grayscale();
      expect(grayscale.size).toEqual(TEST_IMAGES.rgb.size);
      expect(grayscale.channels).toEqual(1);
    });

    it("should convert L to RGB", async () => {
      const rgb = TEST_IMAGES.l.clone().rgb();
      expect(rgb.size).toEqual(TEST_IMAGES.l.size);
      expect(rgb.channels).toEqual(3);
    });

    it("should convert L to RGBA", async () => {
      const rgba = TEST_IMAGES.l.clone().rgba();
      expect(rgba.size).toEqual(TEST_IMAGES.l.size);
      expect(rgba.channels).toEqual(4);
    });

    it("should convert RGB to RGBA", async () => {
      const rgba = TEST_IMAGES.rgb.clone().rgba();
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });

    it("should convert RGBA to RGB", async () => {
      const rgb = TEST_IMAGES.rgba.clone().rgb();
      expect(rgb.size).toEqual(TEST_IMAGES.rgba.size);
      expect(rgb.channels).toEqual(3);
    });
  });

  describe("putAlpha", () => {
    it("should add alpha to RGB image", async () => {
      const rgba = TEST_IMAGES.rgb.clone().putAlpha(TEST_IMAGES.l);
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });
    it("should add alpha to RGBA image", async () => {
      const rgba = TEST_IMAGES.rgba.clone().putAlpha(TEST_IMAGES.l);
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });
  });
});
