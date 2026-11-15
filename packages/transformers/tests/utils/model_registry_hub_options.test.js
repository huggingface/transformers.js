import { env, pipeline, AutoTokenizer, ModelRegistry } from "../../src/transformers.js";

// Files of the fake model repositories used below. Every other file is a 404.
const FILES = {
  "config.json": { model_type: "bert", architectures: ["BertForSequenceClassification"] },
  "tokenizer_config.json": { tokenizer_class: "BertTokenizer" },
};

describe("Hub options in file planning", () => {
  const original = {};
  /** @type {string[]} The revision of each request made. */
  let revisions;

  beforeAll(() => {
    for (const key of ["fetch", "useFSCache", "useBrowserCache", "allowLocalModels"]) {
      original[key] = env[key];
    }
    env.useFSCache = false;
    env.useBrowserCache = false;
    env.allowLocalModels = false;
    env.fetch = async (url) => {
      const [revision, ...path] = String(url).split("/resolve/")[1].split("/");
      revisions.push(revision);
      const file = FILES[path.join("/")];
      if (!file) return new Response("Not Found", { status: 404 });
      const body = JSON.stringify(file);
      return new Response(body, {
        status: 200,
        headers: { "content-type": "application/json", "content-length": String(body.length) },
      });
    };
  });

  beforeEach(() => {
    revisions = [];
  });

  afterAll(() => {
    Object.assign(env, original);
  });

  it("pipeline() only requests the given revision", async () => {
    // The fake repository has no model file, so loading fails after the files have been planned.
    await expect(pipeline("text-classification", "test/pipeline-revision", { revision: "abc123", progress_callback: () => {} })).rejects.toThrow();
    expect(revisions.length).toBeGreaterThan(0);
    expect(new Set(revisions)).toEqual(new Set(["abc123"]));
  });

  it("pipeline() does not make requests with local_files_only", async () => {
    await expect(pipeline("text-classification", "test/pipeline-local", { local_files_only: true, progress_callback: () => {} })).rejects.toThrow();
    expect(revisions).toEqual([]);
  });

  it("AutoTokenizer.from_pretrained() only requests the given revision", async () => {
    // The fake repository has no tokenizer.json.
    await expect(AutoTokenizer.from_pretrained("test/tokenizer-revision", { revision: "abc123" })).rejects.toThrow();
    expect(revisions.length).toBeGreaterThan(0);
    expect(new Set(revisions)).toEqual(new Set(["abc123"]));
  });

  it("ModelRegistry file listings only request the given revision", async () => {
    const options = { revision: "abc123" };
    expect(await ModelRegistry.get_files("test/files-revision", options)).toEqual(["config.json", "onnx/model.onnx", "tokenizer.json", "tokenizer_config.json"]);
    expect(await ModelRegistry.get_pipeline_files("text-classification", "test/pipeline-files-revision", options)).toEqual(["config.json", "onnx/model.onnx", "tokenizer.json", "tokenizer_config.json"]);
    expect(await ModelRegistry.get_tokenizer_files("test/tokenizer-files-revision", options)).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(await ModelRegistry.get_processor_files("test/processor-files-revision", options)).toEqual([]);
    expect(new Set(revisions)).toEqual(new Set(["abc123"]));
  });

  it("uses the main revision by default", async () => {
    await ModelRegistry.get_files("test/files-default");
    expect(new Set(revisions)).toEqual(new Set(["main"]));
  });
});
