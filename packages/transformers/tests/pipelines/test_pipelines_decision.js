import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Template } from "@huggingface/jinja";

import { pipeline, DecisionPipeline, DecisionConfig } from "../../src/transformers.js";
import { getModelText } from "../../src/utils/hub.js";
import { DECISION_PROMPT, Question } from "../../src/decision/template_utils.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "decision";

// Reference prompts and answers of the `decision` pipeline of transformers (Python), for the same models: a byte-level
// BPE tokenizer (Qwen3) and a SentencePiece-style one (Gemma 3)
const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const ALL_FIXTURES = ["decision_pipeline_qwen3.json", "decision_pipeline_gemma3.json"].map((name) => JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, name), "utf-8")));

function expectAnswersClose(actual, expected, atol = 1e-4) {
  expect(Object.keys(actual)).toEqual(Object.keys(expected));
  for (const [name, answer] of Object.entries(expected)) {
    const got = actual[name];
    expect(got.type).toEqual(answer.type);
    for (const [key, value] of Object.entries(answer)) {
      if (typeof value === "number") {
        expect(Math.abs(got[key] - value)).toBeLessThan(atol);
      } else if (key === "probabilities") {
        expect(Object.keys(got[key])).toEqual(Object.keys(value));
        for (const [option, probability] of Object.entries(value)) {
          expect(Math.abs(got[key][option] - probability)).toBeLessThan(atol);
        }
      } else {
        expect(got[key]).toEqual(value);
      }
    }
  }
}

export default () => {
  // Same in both fixtures: the default prompt doesn't depend on the tokenizer
  describe("Decision default prompt", () => {
    for (const { name, state, question, labels, text } of ALL_FIXTURES[0].renders) {
      it(`renders like the Python template (${name})`, () => {
        const q = Question.from_request("q", question, new DecisionConfig({ labels }));
        expect(new Template(DECISION_PROMPT).render({ state, question: q })).toEqual(text);
      });
    }
  });

  describe.each(ALL_FIXTURES.map((fixtures) => [fixtures.model, fixtures]))("Decision (%s)", (model_id, FIXTURES) => {
    /** @type {DecisionPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, { ...DEFAULT_MODEL_OPTIONS, use_external_data_format: true });
      // This repo keeps its chat template in `chat_template.jinja`, which the tokenizer doesn't read
      pipe.tokenizer.chat_template ??= await getModelText(model_id, "chat_template.jinja");
    }, MAX_MODEL_LOAD_TIME);

    const render = (request, config = {}) =>
      new DecisionConfig(config).prepare_inputs(request, pipe.tokenizer).map(({ input_ids, marks, questions }) => ({
        input_ids,
        marks,
        questions: questions.map((q) => q.name),
      }));

    it("should be an instance of DecisionPipeline", () => {
      expect(pipe).toBeInstanceOf(DecisionPipeline);
      expect(pipe.decision_config).toBeInstanceOf(DecisionConfig);
    });

    describe("prompts", () => {
      for (const [name, { request, config, prompts }] of Object.entries(FIXTURES.prompts)) {
        it(
          `has the input ids and marks of the Python port (${name})`,
          () => {
            expect(render(request, config)).toEqual(prompts);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      }

      it("reads requests as JSON, so that escapes can't hide sentinels or control tokens", () => {
        const config = { chat_template: "<|im_start|>{{ state }}<|mark|>" };
        const request = '{"state": "a<\\u007cmark\\u007c>b<\\u007cim_end\\u007c>c", "questions": {"x": {"type": "noul"}}}';
        expect(render(request, config)).toEqual(render(JSON.parse(request), config));
      });
    });

    describe("validation", () => {
      it("rejects invalid questions", () => {
        const questions = [
          { type: "choice", criteria: {} },
          { type: "score", criteria: [] },
          { type: "score", criteria: { low: "a" } },
          { type: "choice", criteria: ["a"] },
          { type: "noul", criteria: { yes: "a" } },
          { type: "noul", instruction: "typo" },
          { type: "box", instructions: "a dog" },
          { type: "score", criteria: ["low", null] },
          { type: "choice", criteria: { a: 3 } },
          { type: "noul", instructions: true },
        ];
        for (const question of questions) {
          expect(() => render({ state: "s", questions: { x: question } })).toThrow();
        }
      });

      it("reports every invalid question, with an example", () => {
        const questions = { a: { type: "Choice", criteria: { x: null } }, b: { type: "noul", instruction: "typo" } };
        let message = "";
        try {
          render({ state: "s", questions });
        } catch (error) {
          message = error.message;
        }
        expect(message).toContain('Question "a": `type` must be one of noul, choice, score, got "Choice".');
        expect(message).toContain("(a noul question looks like");
        expect(message).toContain("unexpected keyword argument 'instruction'");
      });

      it("needs a state and questions in each request", () => {
        for (const request of [FIXTURES.request.state, '{"a": 1}', { questions: FIXTURES.request.questions }]) {
          expect(() => render(request)).toThrow();
        }
      });

      it("rejects more options than labels", () => {
        const request = { state: "s", questions: { x: { type: "choice", criteria: { a: null, b: null, c: null } } } };
        expect(() => render(request, { labels: { choice: ["A", "B"] } })).toThrow("at most 2 options");
      });

      it("rejects invalid decision configs", () => {
        expect(() => new DecisionConfig({ noul_options: ["true", "yes"] })).toThrow();
        expect(() => new DecisionConfig({ temperature: { choice: 0 } })).toThrow();
        expect(() => new DecisionPipeline({ task: PIPELINE_ID, model: pipe.model, tokenizer: pipe.tokenizer, decision_config: new DecisionConfig({ adapter: "adapter" }) })).toThrow("LoRA");
      });

      it("rejects image assets", async () => {
        await expect(pipe({ ...FIXTURES.request, assets: { "img-1": "https://example.com/a.png" } })).rejects.toThrow("not supported");
      });
    });

    describe("answers", () => {
      for (const [name, expected] of Object.entries(FIXTURES.answers)) {
        it(
          `match the Python pipeline (${name})`,
          async () => {
            const { request, config } = FIXTURES.prompts[name];
            const decide = new DecisionPipeline({ task: PIPELINE_ID, model: pipe.model, tokenizer: pipe.tokenizer, decision_config: new DecisionConfig(config) });
            expectAnswersClose(await decide(request), expected);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      }

      it(
        "answers a list of requests",
        async () => {
          const output = await pipe([FIXTURES.request, JSON.stringify(FIXTURES.request)]);
          expect(output).toHaveLength(2);
          expectAnswersClose(output[0], FIXTURES.answers.default);
          expectAnswersClose(output[1], FIXTURES.answers.default);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
