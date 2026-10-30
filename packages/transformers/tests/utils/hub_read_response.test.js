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
});
