import { jest } from "@jest/globals";

import { env } from "../../src/env.js";
import { getFile } from "../../src/utils/hub.js";

describe("Hub cancellation", () => {
  it("passes the loading signal to remote fetches", async () => {
    const originalFetch = env.fetch;
    const controller = new AbortController();
    env.fetch = jest.fn(async (_url, options) => {
      expect(options.signal).toBe(controller.signal);
      return new Response(new Uint8Array());
    });

    try {
      await getFile("https://huggingface.co/test/model", controller.signal);
      expect(env.fetch).toHaveBeenCalledTimes(1);
    } finally {
      env.fetch = originalFetch;
    }
  });
});
