import { env } from "../../src/env.js";
import { getModelFile } from "../../src/utils/hub.js";
import { readResponse } from "../../src/utils/hub/utils.js";

describe("readResponse", () => {
  it("does not pad a short body out to Content-Length", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    const response = new Response(body, {
      headers: { "Content-Length": "8" },
    });

    const out = await readResponse(response, () => {});

    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it("returns a body that matches Content-Length", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    const response = new Response(body, {
      headers: { "Content-Length": "4" },
    });

    const out = await readResponse(response, () => {});

    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it("keeps bytes past a short Content-Length", async () => {
    const body = new Uint8Array([9, 8, 7, 6, 5]);
    const response = new Response(body, {
      headers: { "Content-Length": "2" },
    });

    const out = await readResponse(response, () => {});

    expect(Array.from(out)).toEqual([9, 8, 7, 6, 5]);
  });

  it("does not pad a short body out to expectedSize", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    const response = new Response(body);

    const out = await readResponse(response, () => {}, 8);

    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it("caches the bytes it read and sets Content-Length to that size", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    const stored = [];
    const previous = {
      fetch: env.fetch,
      allowLocalModels: env.allowLocalModels,
      allowRemoteModels: env.allowRemoteModels,
      useCustomCache: env.useCustomCache,
      customCache: env.customCache,
      useBrowserCache: env.useBrowserCache,
      useFSCache: env.useFSCache,
    };
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = false;
    env.useFSCache = false;
    env.useCustomCache = true;
    env.customCache = {
      async match() {
        return undefined;
      },
      async put(_key, response) {
        stored.push(response);
      },
    };
    env.fetch = async () =>
      new Response(body, {
        status: 200,
        headers: { "Content-Length": "8" },
      });
    try {
      const file = await getModelFile("my-org/my-model", "weights.bin", true, {
        progress_callback() {},
      });
      expect(Array.from(file)).toEqual([1, 2, 3, 4]);
      expect(stored).toHaveLength(1);
      expect(stored[0].headers.get("content-length")).toBe("4");
      expect(Array.from(new Uint8Array(await stored[0].arrayBuffer()))).toEqual([1, 2, 3, 4]);
    } finally {
      Object.assign(env, previous);
    }
  });
});
