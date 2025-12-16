import { env } from "../../src/transformers.js";
import { getFile } from "../../src/utils/hub.js";
import { jest } from "@jest/globals";

describe("Hub authorization", () => {
  it("Attaches Authorization header when env.DANGEROUSLY_AVAILABLE_TO_EVERY_USER_HF_TOKEN is set", async () => {
    const originalFetch = global.fetch;
    const mockFetch = jest.fn(() => Promise.resolve(new Response(null)));
    global.fetch = mockFetch;

    const originalHFToken = process.env.HF_TOKEN;
    const originalHFAccessToken = process.env.HF_ACCESS_TOKEN;
    delete process.env.HF_TOKEN;
    delete process.env.HF_ACCESS_TOKEN;

    env.DANGEROUSLY_AVAILABLE_TO_EVERY_USER_HF_TOKEN = "hf_dummy";

    await getFile("https://huggingface.co/any/model");

    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers.get("Authorization")).toBe("Bearer hf_dummy");

    env.DANGEROUSLY_AVAILABLE_TO_EVERY_USER_HF_TOKEN = null;
    global.fetch = originalFetch;
    if (originalHFToken !== undefined) process.env.HF_TOKEN = originalHFToken; else delete process.env.HF_TOKEN;
    if (originalHFAccessToken !== undefined) process.env.HF_ACCESS_TOKEN = originalHFAccessToken; else delete process.env.HF_ACCESS_TOKEN;
  });
});
