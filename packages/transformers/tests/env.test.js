import { env } from "../src/transformers.js";

describe("env.remoteHost", () => {
  const originalEnvEndpoint = process.env.HF_ENDPOINT;

  afterEach(() => {
    if (originalEnvEndpoint !== undefined) {
      process.env.HF_ENDPOINT = originalEnvEndpoint;
    } else {
      delete process.env.HF_ENDPOINT;
    }
    env.remoteHost = undefined;
  });

  it("defaults to the Hugging Face Hub when HF_ENDPOINT is unset", () => {
    delete process.env.HF_ENDPOINT;
    env.remoteHost = undefined;
    expect(env.remoteHost).toBe("https://huggingface.co/");
  });

  it("treats an empty HF_ENDPOINT environment variable as unset", () => {
    process.env.HF_ENDPOINT = "";
    env.remoteHost = undefined;
    expect(env.remoteHost).toBe("https://huggingface.co/");
  });

  it("treats a whitespace-only HF_ENDPOINT environment variable as unset", () => {
    process.env.HF_ENDPOINT = "   ";
    env.remoteHost = undefined;
    expect(env.remoteHost).toBe("https://huggingface.co/");
  });

  it("uses custom HF_ENDPOINT environment variable when set", () => {
    process.env.HF_ENDPOINT = "https://custom.endpoint.com/";
    env.remoteHost = undefined;
    expect(env.remoteHost).toBe("https://custom.endpoint.com/");
  });

  it("preserves explicit programmatic empty string override", () => {
    process.env.HF_ENDPOINT = "https://custom.endpoint.com/";
    env.remoteHost = "";
    expect(env.remoteHost).toBe("");
  });

  it("restores fallback behavior when assigning undefined", () => {
    process.env.HF_ENDPOINT = "https://custom.endpoint.com/";
    env.remoteHost = "https://manual.override.com/";
    expect(env.remoteHost).toBe("https://manual.override.com/");

    env.remoteHost = undefined;
    expect(env.remoteHost).toBe("https://custom.endpoint.com/");

    delete process.env.HF_ENDPOINT;
    expect(env.remoteHost).toBe("https://huggingface.co/");
  });
});
