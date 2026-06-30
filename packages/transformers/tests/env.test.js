import { spawnSync } from "child_process";

const ENV_MODULE_URL = new URL("../src/env.js", import.meta.url).href;

function readRemoteHost(env) {
  const code = `import { env } from ${JSON.stringify(ENV_MODULE_URL)}; process.stdout.write(env.remoteHost);`;
  const result = spawnSync("node", ["--input-type=module", "-e", code], { env });

  expect(result.stderr.toString()).toEqual("");
  expect(result.status).toEqual(0);

  return result.stdout.toString();
}

describe("env", () => {
  it("defaults remoteHost to the Hugging Face Hub", async () => {
    const env = { ...process.env };
    delete env.HF_ENDPOINT;

    expect(readRemoteHost(env)).toBe("https://huggingface.co/");
  });

  it("uses HF_ENDPOINT as the default remoteHost when set", async () => {
    const env = { ...process.env, HF_ENDPOINT: "https://hf.example.com/" };

    expect(readRemoteHost(env)).toBe("https://hf.example.com/");
  });
});
