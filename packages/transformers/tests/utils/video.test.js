import { jest } from "@jest/globals";

const scopedFetch = jest.fn(async () => ({ blob: async () => new Blob([]) }));

jest.unstable_mockModule("../../src/env.js", () => ({
  apis: { IS_BROWSER_ENV: true },
  resolveEnv: (sessionEnv = {}) => ({ fetch: sessionEnv.fetch ?? globalThis.fetch }),
}));

jest.unstable_mockModule("../../src/utils/image.js", () => ({
  RawImage: class RawImage {
    constructor(data, width, height, channels) {
      this.data = data;
      this.width = width;
      this.height = height;
      this.channels = channels;
    }
  },
}));

const { load_video } = await import("../../src/utils/video.js");

describe("Video session env", () => {
  const originalDocument = globalThis.document;
  const originalURL = globalThis.URL;

  beforeEach(() => {
    scopedFetch.mockClear();

    const video = {
      crossOrigin: "",
      muted: false,
      src: "https://example.com/video.mp4",
      duration: 1,
      videoWidth: 1,
      videoHeight: 1,
      seekable: { start: () => 0, end: () => 0 },
      remove: jest.fn(),
      set onloadedmetadata(callback) {
        queueMicrotask(callback);
      },
      set onseeked(callback) {
        queueMicrotask(callback);
      },
    };
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage: () => {},
        getImageData: () => ({ data: new Uint8ClampedArray(4) }),
      }),
    };

    globalThis.document = {
      createElement: (type) => (type === "video" ? video : canvas),
    };
    globalThis.URL = class extends originalURL {
      static createObjectURL() {
        return "blob:test";
      }
    };
  });

  afterAll(() => {
    globalThis.document = originalDocument;
    globalThis.URL = originalURL;
  });

  it("should use options.env.fetch when downloading a non-seekable video", async () => {
    await load_video("https://example.com/video.mp4", {
      num_frames: 1,
      env: { fetch: scopedFetch },
    });

    expect(scopedFetch).toHaveBeenCalledWith("https://example.com/video.mp4");
  });
});
