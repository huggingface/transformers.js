import { parse_response, ResponseParser } from "../../src/utils/chat_parsing/index.js";

// Mirrors transformers/tests/utils/test_chat_parsing.py

const cohere_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
  fields: {
    content: { open: "<|START_RESPONSE|>", close: "<|END_RESPONSE|>", content: "text" },
    thinking: { open: "<|START_THINKING|>", close: "<|END_THINKING|>", content: "text" },
    tool_calls: {
      open: "<|START_ACTION|>",
      close: "<|END_ACTION|>",
      content: "json",
      transform_each: true,
      transform: { type: "function", function: { name: "{tool_name}", arguments: "{parameters}" } },
    },
  },
};

const ernie_template = {
  defaults: { role: "assistant" },
  start_anchor: "Assistant:",
  fields: {
    thinking: { open_pattern: String.raw`(?:^|<think>\s*)`, close: "</think>", content: "text" },
    content: { open: "<response>\n", close_pattern: String.raw`\n?</response>`, content: "text" },
    tool_calls: {
      open: "<tool_call>",
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
  },
};

const gpt_oss_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|start|>assistant",
  fields: {
    thinking: { open: "<|channel|>analysis<|message|>", close: "<|end|>", content: "text" },
    content: { open: "<|channel|>final<|message|>", close: ["<|end|>", "<|return|>"], content: "text" },
    tool_calls: {
      open_pattern: String.raw`<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>`,
      close: "<|call|>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
    },
  },
};

const smollm_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<tool_call>",
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
    content: { close: "<|im_end|>", content: "text" },
  },
};

const qwen3_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open_pattern: String.raw`<tool_call>\s*<function=(?P<name>\w+)>`,
      close: "</tool_call>",
      repeats: true,
      content: "xml-inline",
      content_args: {
        tag_pattern: String.raw`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>`,
        value_parser: { name: "json", args: { allow_non_json: true } },
      },
      transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
    },
  },
};

const gemma4_template = {
  defaults: { role: "assistant" },
  // The chat template only emits `<|turn>model\n` when the previous message wasn't a tool_call/
  // tool_response. After a tool_response the prefix just ends with `<tool_response|>` and the
  // model continues from there, so we accept either anchor and truncate past the latest one.
  start_anchor: ["<|turn>model\n", "<tool_response|>"],
  fields: {
    thinking: { open: "<|channel>thought\n", close: "<channel|>", content: "text" },
    tool_calls: {
      open_pattern: String.raw`<\|tool_call>call:(?P<name>\w+)`,
      close: "<tool_call|>",
      repeats: true,
      content: "json",
      content_args: { unquoted_keys: true, string_delims: [['<|"|>', '<|"|>']] },
      transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
    },
    content: { close: ["<turn|>", "<|tool_response>", "<eos>"], content: "text" },
  },
};

const granite30_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|start_of_role|>assistant<|end_of_role|>",
  fields: {
    tool_calls: {
      open_pattern: String.raw`\{\s*(?="tool"\s*:)`,
      close: "<|end_of_text|>",
      content: "json",
      content_args: { prefix: "{" },
      transform: [{ type: "function", function: { name: "{content.tool}", arguments: "{content.parameters}" } }],
    },
    content: { close: "<|end_of_text|>", content: "text" },
  },
};

const granite33_template = {
  defaults: { role: "assistant" },
  start_anchor_pattern: String.raw`<\|start_of_role\|>assistant(?:\s+\[[^\n]*\])?<\|end_of_role\|>`,
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<|tool_call|>",
      close: "<|end_of_text|>",
      content: "json",
      transform_each: true,
      transform: { type: "function", function: { name: "{name}", arguments: "{arguments}" } },
    },
    content: {
      close: "<|end_of_text|>",
      content: "text",
      content_args: { strip_prefix: "<response>", strip_suffix: "</response>" },
    },
  },
};

const granite4_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|start_of_role|>assistant<|end_of_role|>",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<tool_call>",
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
    content: { close: "<|end_of_text|>", content: "text" },
  },
};

const lfm_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open: "<|tool_call_start|>",
      close: "<|tool_call_end|>",
      content: "python-call-list",
      transform_each: true,
      transform: { type: "function", function: { name: "{name}", arguments: "{arguments}" } },
    },
    content: { close: "<|im_end|>", content: "text" },
  },
};

// Inkling (TMLv0) frames every block as <|message_model|>[author-name]<|content_KIND|>body<|end_message|>;
// the generation prompt pre-writes the first <|message_model|>, so each open treats the header as optional.
const inkling_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|message_model|>",
  fields: {
    thinking: {
      open_pattern: String.raw`(?:<\|message_model\|>)?[^<]*<\|content_thinking\|>`,
      close: "<|end_message|>",
      repeats: true,
      join: "",
      content_args: { strip: false },
    },
    content: {
      open_pattern: String.raw`(?:<\|message_model\|>)?[^<]*<\|content_text\|>`,
      close: "<|end_message|>",
      repeats: true,
      join: "",
      content_args: { strip: false },
    },
    tool_calls: {
      open_pattern: String.raw`(?:<\|message_model\|>)?[^<]*<\|content_invoke_tool_json\|>`,
      close: "<|end_message|>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: { name: "{content.name}", arguments: "{content.args}" } },
    },
  },
};

// Fallback template used by `transformers serve` for Qwen models. Its `\s*`-prefixed patterns
// have no literal prefix, so streaming relies on true partial matching.
const qwen_serve_template = {
  defaults: { role: "assistant" },
  start_anchor: "<|im_start|>assistant\n",
  fields: {
    thinking: { open: "<think>", close: "</think>", content: "text" },
    tool_calls: {
      open_pattern: String.raw`\s*<tool_call>`,
      close: "</tool_call>",
      repeats: true,
      content: "json",
      transform: { type: "function", function: "{content}" },
    },
    content: { close_pattern: String.raw`\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)`, content: "text" },
  },
};

const COHERE_OUTPUT = "<|START_THINKING|>I should call a tool.<|END_THINKING|>" + '<|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", ' + '"parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>';

// Fixtures shared by the streaming tests: one representative input per template.
const STREAMING_FIXTURES = [
  ["cohere", cohere_template, "<|START_THINKING|>I should call a tool.<|END_THINKING|>" + '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "simple_tool", ' + '"parameters": {"a": 1}}]<|END_ACTION|>'],
  ["ernie", ernie_template, "<think>some deliberation here</think>\n\n" + '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>'],
  ["gpt_oss", gpt_oss_template, "<|channel|>analysis<|message|>thinking chunk<|end|><|channel|>final<|message|>done text"],
  [
    // The `tool_calls` open_pattern's full match spans far more than a small fixed hold window.
    "gpt_oss_tool",
    gpt_oss_template,
    "<|channel|>analysis<|message|>Let me check.<|end|>" + "<|start|>assistant<|channel|>commentary to=functions.get_current_weather " + '<|constrain|>json<|message|>{"location": "San Francisco, CA"}<|call|>',
  ],
  ["smollm", smollm_template, '<think>thinking</think>\n<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>'],
  ["qwen3", qwen3_template, "<think>short thought</think>\n" + "<tool_call>\n<function=get_weather>\n" + "<parameter=city>\nParis\n</parameter>\n" + "</function>\n</tool_call>"],
  ["gemma4", gemma4_template, '<|channel>thought\nhi<channel|><|tool_call>call:foo{a:1,b:<|"|>bar<|"|>}<tool_call|>'],
  ["granite30", granite30_template, '{"tool":"get_weather","parameters":{"city":"Paris"}}<|end_of_text|>'],
  ["granite33", granite33_template, "<think>check weather</think><response>It is sunny.</response><|end_of_text|>"],
  ["granite4", granite4_template, '<think>check weather</think><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call><|end_of_text|>'],
  ["lfm", lfm_template, '<think>check weather</think><|tool_call_start|>[get_weather(city="Paris", options={"units": "C"})]<|tool_call_end|><|im_end|>'],
  [
    // Exercises `join` fields (two thinking blocks) and dotted transform paths.
    "inkling",
    inkling_template,
    "<|content_thinking|>Consider the weather.<|end_message|>" + "<|message_model|><|content_thinking|> Tokyo, probably.<|end_message|>" + "<|message_model|><|content_text|>Checking now.<|end_message|>" + "<|message_model|>get_weather<|content_invoke_tool_json|>" + '{"name":"get_weather","args":{"city":"Tokyo"}}<|end_message|>' + "<|content_model_end_sampling|>",
  ],
  ["qwen_serve", qwen_serve_template, "<think>\nhmm\n</think>\n\nSure.\n<tool_call>\n" + '{"name": "fn", "arguments": {"x": 1}}\n</tool_call>\n<|im_end|>\n<|endoftext|>'],
];

function* chunk_fixed(text, step) {
  for (let i = 0; i < text.length; i += step) yield text.slice(i, i + step);
}

/** Seeded PRNG (mulberry32), so failures reproduce. */
function make_rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Split `text` into a random number of non-empty chunks at random cut points. */
function* chunk_random(text, rng) {
  let previous = 0;
  for (let i = 1; i < text.length; ++i) {
    if (rng() < 0.5) {
      yield text.slice(previous, i);
      previous = i;
    }
  }
  yield text.slice(previous);
}

function stream_all(template, chunks, options = { prefix: "" }) {
  const parser = new ResponseParser(template, options);
  const events = [...parser.initial_events];
  for (const chunk of chunks) events.push(...parser.feed(chunk));
  const [message, final_events] = parser.finalize();
  return { message, events: [...events, ...final_events] };
}

describe("Response templates", () => {
  describe("Parsing", () => {
    it.each([
      '{location: "London", unit: "celsius"}',
      '{\n  location : "London",\n  unit : "celsius"\n}',
    ])("gemma4 tool arguments with whitespace: %s", (argumentsText) => {
      const model_out = `<|tool_call>call:get_weather${argumentsText}<tool_call|>`;
      const expected = {
        role: "assistant",
        tool_calls: [{ type: "function", function: { name: "get_weather", arguments: { location: "London", unit: "celsius" } } }],
      };
      expect(parse_response(model_out, gemma4_template, { prefix: "" })).toEqual(expected);
      expect(stream_all(gemma4_template, chunk_fixed(model_out, 1)).message).toEqual(expected);
    });

    it("cohere", () => {
      expect(parse_response(COHERE_OUTPUT, cohere_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "I should call a tool.",
        tool_calls: [{ type: "function", function: { name: "simple_tool", arguments: { temperature_format: "Celsius" } } }],
      });
    });

    it("ernie with tools", () => {
      const model_out = "The user is asking about the weather in Paris today. Let me check the available tools.\n" + "</think>\n\n" + '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>';
      expect(parse_response(model_out, ernie_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "The user is asking about the weather in Paris today. Let me check the available tools.",
        tool_calls: [{ type: "function", function: { name: "get_current_temperature", arguments: { location: "Paris" } } }],
      });
    });

    it("ernie without tools", () => {
      const model_out = 'The user just greeted me with "Hi! How are you?"\n\nKeep the tone warm.\n' + "</think>\n\n" + "<response>\nHello! I'm doing well, thank you for asking.\n</response>\n</s>";
      expect(parse_response(model_out, ernie_template, { prefix: "" })).toEqual({
        role: "assistant",
        content: "Hello! I'm doing well, thank you for asking.",
        thinking: 'The user just greeted me with "Hi! How are you?"\n\nKeep the tone warm.',
      });
    });

    it("gpt-oss with tool call", () => {
      const model_out = "<|channel|>analysis<|message|>We need to call get_current_weather.<|end|>" + "<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>" + '{\n  "location": "San Francisco, CA"\n}';
      expect(parse_response(model_out, gpt_oss_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "We need to call get_current_weather.",
        tool_calls: [{ type: "function", function: { name: "get_current_weather", arguments: { location: "San Francisco, CA" } } }],
      });
    });

    it("gpt-oss without tool call", () => {
      const model_out = "<|channel|>analysis<|message|>User asks a simple math question: 2+2 = 4. Provide answer." + "<|end|><|start|>assistant<|channel|>final<|message|>2";
      expect(parse_response(model_out, gpt_oss_template, { prefix: "" })).toEqual({
        role: "assistant",
        content: "2",
        thinking: "User asks a simple math question: 2+2 = 4. Provide answer.",
      });
    });

    it("smollm thinking and tool call", () => {
      const model_out = '<think>\nOkay, the user said, "Hello! How are you?"\n</think>\n\n' + '<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hello!"}}</tool_call>';
      expect(parse_response(model_out, smollm_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: 'Okay, the user said, "Hello! How are you?"',
        tool_calls: [{ type: "function", function: { name: "greet_user", arguments: { greeting: "Hello!" } } }],
      });
    });

    it("smollm tool call without thinking", () => {
      const model_out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>';
      expect(parse_response(model_out, smollm_template, { prefix: "" })).toEqual({
        role: "assistant",
        tool_calls: [{ type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }],
      });
    });

    it("smollm thinking without tool call", () => {
      const model_out = "<think>\nLet me explain gravity.</think>\nSome content about gravity.";
      expect(parse_response(model_out, smollm_template, { prefix: "" })).toEqual({
        role: "assistant",
        content: "Some content about gravity.",
        thinking: "Let me explain gravity.",
      });
    });

    it("qwen3 tool calls", () => {
      const model_out = "<tool_call>\n<function=get_weather>\n<parameter=locations>\n" + '[{"country": "France", "city": "Paris"}]\n</parameter>\n' + "<parameter=temp_units>\ncelsius\n</parameter>\n</function>\n</tool_call>";
      expect(parse_response(model_out, qwen3_template, { prefix: "" })).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_weather",
              arguments: { locations: [{ country: "France", city: "Paris" }], temp_units: "celsius" },
            },
          },
        ],
      });
    });

    it("gemma4 tool call", () => {
      const model_out = "<|channel>thought\nI should check the available tools.<channel|>" + '<|tool_call>call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,' + 'unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>';
      expect(parse_response(model_out, gemma4_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "I should check the available tools.",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_current_temperature",
              arguments: { detail_level: 0, location: "Paris, France", unit: "celsius" },
            },
          },
        ],
      });
    });

    it("gemma4 complex tool call", () => {
      const model_out = "<|channel>thought\nLet me call the tool.<channel|>" + '<|tool_call>call:foo{bool_value:true,list_value:[<|"|>foo<|"|>,<|"|>bar<|"|>],' + 'null_value:null,number_value:1,string_value:<|"|>foo<|"|>,' + 'struct_value:{foo:<|"|>bar<|"|>}}<tool_call|>';
      expect(parse_response(model_out, gemma4_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "Let me call the tool.",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "foo",
              arguments: {
                bool_value: true,
                list_value: ["foo", "bar"],
                null_value: null,
                number_value: 1,
                string_value: "foo",
                struct_value: { foo: "bar" },
              },
            },
          },
        ],
      });
    });

    it("granite 3.0 tool call", () => {
      const model_out = '{\n    "tool": "get_weather",\n    "parameters": {\n        "city": "Paris"\n    }\n}<|end_of_text|>';
      expect(parse_response(model_out, granite30_template, { prefix: "" })).toEqual({
        role: "assistant",
        tool_calls: [{ type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }],
      });
    });

    it("granite 3.3 tool calls", () => {
      const model_out = '<|tool_call|>[{"name":"get_weather","arguments":{"cities":["Paris","Tokyo"]}},{"name":"get_time","arguments":{"utc":true}}]<|end_of_text|>';
      expect(parse_response(model_out, granite33_template, { prefix: "" })).toEqual({
        role: "assistant",
        tool_calls: [
          { type: "function", function: { name: "get_weather", arguments: { cities: ["Paris", "Tokyo"] } } },
          { type: "function", function: { name: "get_time", arguments: { utc: true } } },
        ],
      });
    });

    it("granite 3 thinking response", () => {
      const model_out = "<think>I should answer briefly.</think><response>Paris.</response><|end_of_text|>";
      expect(parse_response(model_out, granite33_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "I should answer briefly.",
        content: "Paris.",
      });
    });

    it("granite 4 tool calls", () => {
      const model_out = '<think>Use two tools.</think><tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>\n<tool_call>{"name":"get_time","arguments":{"utc":true}}</tool_call><|end_of_text|>';
      expect(parse_response(model_out, granite4_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "Use two tools.",
        tool_calls: [
          { type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } },
          { type: "function", function: { name: "get_time", arguments: { utc: true } } },
        ],
      });
    });

    it("lfm pythonic tool calls", () => {
      const model_out = '<think>Use two tools.</think><|tool_call_start|>[get_weather(city="Paris", options={"units": "C", "days": [1, 2]}), set_alarm(hour=7, enabled=True, note=None)]<|tool_call_end|>Checking now.<|im_end|>';
      expect(parse_response(model_out, lfm_template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "Use two tools.",
        tool_calls: [
          { type: "function", function: { name: "get_weather", arguments: { city: "Paris", options: { units: "C", days: [1, 2] } } } },
          { type: "function", function: { name: "set_alarm", arguments: { hour: 7, enabled: true, note: null } } },
        ],
        content: "Checking now.",
      });
    });

    it("rejects malformed lfm pythonic tool calls without evaluating them", () => {
      const model_out = "<|tool_call_start|>[get_weather(__proto__=process.exit())]<|tool_call_end|>";
      expect(() => parse_response(model_out, lfm_template, { prefix: "" })).toThrow("unsupported value");
    });

    it("parses python string escapes and accepts trailing commas in lfm tool calls", () => {
      const model_out = String.raw`<|tool_call_start|>[search(query="a\q", quote="a\"b", path="a\\b", octal="\101", options={"limit": 2,},),]<|tool_call_end|>`;
      expect(parse_response(model_out, lfm_template, { prefix: "" }).tool_calls).toEqual([
        {
          type: "function",
          function: {
            name: "search",
            arguments: {
              query: String.raw`a\q`,
              quote: 'a"b',
              path: String.raw`a\b`,
              octal: "A",
              options: { limit: 2 },
            },
          },
        },
      ]);
    });

    it("handles python line continuations and rejects named unicode escapes", () => {
      const continued = '<|tool_call_start|>[search(query="a\\\nb")]<|tool_call_end|>';
      expect(parse_response(continued, lfm_template, { prefix: "" }).tool_calls[0].function.arguments.query).toBe("ab");
      const named = String.raw`<|tool_call_start|>[search(query="\N{LATIN CAPITAL LETTER A}")]<|tool_call_end|>`;
      expect(() => parse_response(named, lfm_template, { prefix: "" })).toThrow("named unicode escapes are not supported");
    });

    it("treats empty text wrappers as no-ops", () => {
      const template = {
        start_anchor: "<assistant>",
        fields: { content: { content: "text", content_args: { strip_prefix: "", strip_suffix: "" } } },
      };
      expect(parse_response("answer", template, { prefix: "" })).toEqual({ content: "answer" });
      expect(stream_all(template, ["answer"]).events.find((event) => event.type === "region_chunk")?.dirty).toBe(false);
    });

    it("inkling multi-block message", () => {
      const model_out = "<|content_thinking|>Consider the weather.<|end_message|>" + "<|message_model|><|content_thinking|> Tokyo, probably.<|end_message|>" + "<|message_model|><|content_text|>Checking the weather now.<|end_message|>" + "<|message_model|>get_weather<|content_invoke_tool_json|>" + '{"name":"get_weather","args":{"city":"Tokyo","units":"C"}}<|end_message|>' + "<|content_model_end_sampling|>";
      const prefix = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_model|>";
      expect(parse_response(model_out, inkling_template, { prefix })).toEqual({
        role: "assistant",
        thinking: "Consider the weather. Tokyo, probably.",
        content: "Checking the weather now.",
        tool_calls: [{ type: "function", function: { name: "get_weather", arguments: { city: "Tokyo", units: "C" } } }],
      });
    });

    it("transform dotted paths", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: {
          tool_calls: {
            open: "<tool>",
            close: "</tool>",
            repeats: true,
            content: "json",
            transform: { type: "function", function: { name: "{content.name}", arguments: "{content.args}" } },
          },
        },
      };
      const model_out = '<tool>{"name": "get_weather", "args": {"city": {"id": 7}}}</tool>';
      expect(parse_response(model_out, template, { prefix: "" }).tool_calls).toEqual([{ type: "function", function: { name: "get_weather", arguments: { city: { id: 7 } } } }]);
    });

    it("transform dotted paths with transform_each", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: {
          tool_calls: {
            open: "<actions>",
            close: "</actions>",
            content: "json",
            transform_each: true,
            transform: { type: "function", function: { name: "{fn.name}", arguments: "{fn.args}" } },
          },
        },
      };
      const model_out = '<actions>[{"fn": {"name": "a", "args": {"x": 1}}}, {"fn": {"name": "b", "args": {}}}]</actions>';
      expect(parse_response(model_out, template, { prefix: "" }).tool_calls).toEqual([
        { type: "function", function: { name: "a", arguments: { x: 1 } } },
        { type: "function", function: { name: "b", arguments: {} } },
      ]);
    });

    it("transform dotted path errors", () => {
      const spec_with = (transform) => ({
        start_anchor: "<|assistant|>",
        fields: { x: { open: "<x>", close: "</x>", content: "json", transform } },
      });
      expect(() => parse_response('<x>{"name": "n"}</x>', spec_with({ a: "{content.args}" }), { prefix: "" })).toThrow("missing key 'args'");
      expect(() => parse_response('<x>{"name": "n"}</x>', spec_with({ a: "{content.name.x}" }), { prefix: "" })).toThrow("cannot index into string");
      expect(() => parse_response("<x>{}</x>", spec_with({ a: "{missing}" }), { prefix: "" })).toThrow("is not defined");
    });

    it("transform mixing a dotted placeholder with literal text is rejected", () => {
      const template = {
        start_anchor: "<|assistant|>",
        fields: { x: { open: "<x>", close: "</x>", transform: { v: "pre {content.args}" } } },
      };
      expect(() => parse_response("", template, { prefix: "" })).toThrow("mixes");
    });

    it("join concatenates repeated matches", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: {
          thinking: { open: "<think>", close: "</think>", repeats: true, join: " " },
          content: { repeats: true, join: " " },
        },
      };
      expect(parse_response("<think>first</think>middle<think>second</think>done", template, { prefix: "" })).toEqual({
        role: "assistant",
        thinking: "first second",
        content: "middle done",
      });
      expect(parse_response("<think>only</think>", template, { prefix: "" })).toEqual({ role: "assistant", thinking: "only" });
    });

    it("join validation", () => {
      const no_repeats = { start_anchor: "a", fields: { x: { open: "<x>", close: "</x>", join: "" } } };
      expect(() => parse_response("", no_repeats, { prefix: "" })).toThrow("requires 'repeats'");
      const bad_type = { start_anchor: "a", fields: { x: { open: "<x>", close: "</x>", repeats: true, join: 7 } } };
      expect(() => parse_response("", bad_type, { prefix: "" })).toThrow("must be a string");
    });

    it("join requires string matches", () => {
      const template = {
        start_anchor: "a",
        fields: { x: { open: "<x>", close: "</x>", repeats: true, join: "", content: "json" } },
      };
      expect(() => parse_response("<x>{}</x>", template, { prefix: "" })).toThrow("parse to a string");
    });

    it("optional: false raises when missing", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { content: { open: "<response>", close: "</response>", content: "text", optional: false } },
      };
      expect(() => parse_response("no response here", template, { prefix: "" })).toThrow('["content"]');
    });

    it("int content parser", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { count: { open: "<n>", close: "</n>", content: "int" } },
      };
      expect(parse_response("<n>42</n>", template, { prefix: "" })).toEqual({ role: "assistant", count: 42 });
    });

    it("kv-lines parser", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { metadata: { open: "<meta>", close: "</meta>", content: "kv-lines" } },
      };
      expect(parse_response("<meta>name: alice\nage: 30</meta>", template, { prefix: "" })).toEqual({
        role: "assistant",
        metadata: { name: "alice", age: "30" },
      });
    });

    it("rejects invalid templates", () => {
      const template = (fields, extra = {}) => ({ defaults: { role: "assistant" }, start_anchor: "<|assistant|>", fields, ...extra });
      expect(() => parse_response("[hi]", template({ x: { open: "[", close: "]", content: "not-a-real-parser" } }))).toThrow("unknown content parser");
      expect(() => parse_response("hello", template({ content: { content: "text" } }, { version: 2 }))).toThrow("version");
      expect(() => parse_response("hello", template({ a: { content: "text" }, b: { content: "text" } }))).toThrow("At most one field");
      expect(() => parse_response("<tool>foo</tool>", template({ tool: { open: "<tool>", close: "</tool>", content: "text", transform: { label: "name: {content}" } } }))).toThrow(/\{content\}.*interpolation/);
      expect(() => parse_response("<tool name=foo>body</tool>", template({ tool: { open_pattern: String.raw`<tool name=(?P<name>\w+)>`, close: "</tool>", content: "text" } }))).toThrow(/\["name"\].*'transform'/);
    });

    it("literal list open and close", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { x: { open: ["<a>", "<bb>"], close: ["</a>", "</bb>"], content: "text" } },
      };
      for (const [opener, closer] of [
        ["<a>", "</a>"],
        ["<bb>", "</bb>"],
        ["<a>", "</bb>"],
      ]) {
        expect(parse_response(`${opener}hi${closer}`, template, { prefix: "" })).toEqual({ role: "assistant", x: "hi" });
      }
    });

    it("literal list streams without holding back unrelated text", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { content: { close: ["<turn|>", "<|tool_response>", "<eos>"], content: "text" } },
      };
      const parser = new ResponseParser(template, { prefix: "" });
      const plain = "x".repeat(32);
      const flushed = parser
        .feed(plain)
        .filter((e) => e.type === "region_chunk")
        .map((e) => e.text);
      expect(flushed.join("")).toEqual(plain);
    });

    it("literal list defers a prefix-overlapping literal", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { x: { open: "<x>", close: ["END", "ENDX"], content: "text" } },
      };
      const parser = new ResponseParser(template, { prefix: "" });
      // "<x>hiEND" mid-stream: don't commit the close yet — "ENDX" might be coming.
      const events = parser.feed("<x>hiEND");
      expect(events.filter((e) => e.type === "region_close")).toEqual([]);
      // Once a non-matching character arrives, the deferred close commits with the shorter literal.
      events.push(...parser.feed(" more"));
      const [message] = parser.finalize();
      const closes = events.filter((e) => e.type === "region_close" && e.field === "x");
      expect(closes).toEqual([{ type: "region_close", field: "x", value: "hi" }]);
      expect(message).toEqual({ role: "assistant", x: "hi" });
    });

    it("literal list rejects empty and non-string literals", () => {
      for (const bad_open of [[], [""], [1, 2], { foo: "bar" }]) {
        const template = {
          defaults: { role: "assistant" },
          start_anchor: "<|assistant|>",
          fields: { x: { open: bad_open, close: "</x>", content: "text" } },
        };
        expect(() => parse_response("<x>hi</x>", template, { prefix: "" })).toThrow();
      }
    });

    it("field without close runs to end of stream", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: { content: { open: "<resp>", content: "text" } },
      };
      expect(parse_response("<resp>hello world", template, { prefix: "" })).toEqual({ role: "assistant", content: "hello world" });
    });
  });

  describe("Streaming", () => {
    it("matches the whole-string parse for fixed chunk sizes", () => {
      for (const [, template, text] of STREAMING_FIXTURES) {
        const expected = parse_response(text, template, { prefix: "" });
        for (const step of [1, 2, 3, 5, 7, 13, 31]) {
          expect(stream_all(template, chunk_fixed(text, step)).message).toEqual(expected);
        }
      }
    });

    it("matches the whole-string parse for random chunkings", () => {
      const rng = make_rng(0xc0de5eed);
      for (const [, template, text] of STREAMING_FIXTURES) {
        const expected = parse_response(text, template, { prefix: "" });
        for (let trial = 0; trial < 30; ++trial) {
          expect(stream_all(template, chunk_random(text, rng)).message).toEqual(expected);
        }
      }
    });

    it("emits well-formed events for every chunking", () => {
      const rng = make_rng(0xbeef);
      for (const [, template, text] of STREAMING_FIXTURES) {
        for (let trial = 0; trial < 10; ++trial) {
          let open_field = null;
          for (const event of stream_all(template, chunk_random(text, rng)).events) {
            if (event.type === "region_open") {
              expect(open_field).toBeNull();
              open_field = event.field;
            } else if (event.type === "region_chunk") {
              expect(event.field).toBe(open_field);
              expect(typeof event.dirty).toBe("boolean");
            } else {
              expect(event.type).toBe("region_close");
              expect(event.field).toBe(open_field);
              open_field = null;
            }
          }
          expect(open_field).toBeNull();
        }
      }
    });

    it("streams text regions as they are generated", () => {
      // Every character of a text region is emitted as soon as it can no longer be part of a delimiter.
      const [, template, text] = STREAMING_FIXTURES[0];
      const { events } = stream_all(template, chunk_fixed(text, 1));
      const chunks = (field) =>
        events
          .filter((e) => e.type === "region_chunk" && e.field === field)
          .map((e) => e.text)
          .join("");
      expect(chunks("thinking")).toEqual("I should call a tool.");
      expect(events.find((e) => e.type === "region_close" && e.field === "thinking").value).toEqual("I should call a tool.");
      // `tool_calls` is json → dirty chunks stream the raw body, parsed value on close.
      expect(chunks("tool_calls")).toEqual('[{"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"a": 1}}]');
      expect(new Set(events.filter((e) => e.type === "region_chunk").map((e) => `${e.field}:${e.dirty}`))).toEqual(new Set(["thinking:false", "tool_calls:true"]));
    });

    it("marks structured regions as dirty", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|assistant|>",
        fields: {
          thinking: { open: "<t>", close: "</t>", content: "text" },
          score: { open: "<n>", close: "</n>", content: "int" },
          json_call: { open: "<j>", close: "</j>", content: "json" },
          xml_call: {
            open: "<x>",
            close: "</x>",
            content: "xml-inline",
            content_args: { tag_pattern: String.raw`<(?P<key>\w+)=(?P<value>[^>]+)>` },
          },
          kv_call: { open: "<kv>", close: "</kv>", content: "kv-lines" },
        },
      };
      const text = '<t>hello world</t><n>42</n><j>{"a": 1, "b": 2}</j><x><name=foo><age=10></x><kv>k1: v1\nk2: v2</kv>';
      const { events } = stream_all(template, chunk_fixed(text, 1));
      const chunks = {};
      const dirty = {};
      for (const e of events.filter((e) => e.type === "region_chunk")) {
        chunks[e.field] = (chunks[e.field] ?? "") + e.text;
        (dirty[e.field] ??= new Set()).add(e.dirty);
      }
      for (const field of ["thinking", "score"]) expect(dirty[field]).toEqual(new Set([false]));
      for (const field of ["json_call", "xml_call", "kv_call"]) expect(dirty[field]).toEqual(new Set([true]));
      expect(chunks).toEqual({
        thinking: "hello world",
        score: "42",
        json_call: '{"a": 1, "b": 2}',
        xml_call: "<name=foo><age=10>",
        kv_call: "k1: v1\nk2: v2",
      });
    });

    it("marks wrapper-stripped text as dirty until close", () => {
      const { message, events } = stream_all(granite33_template, chunk_fixed("<think>reason</think><response>answer</response><|end_of_text|>", 1));
      expect(message).toEqual({ role: "assistant", thinking: "reason", content: "answer" });
      expect(new Set(events.filter((e) => e.type === "region_chunk" && e.field === "content").map((e) => e.dirty))).toEqual(new Set([true]));
    });

    it("streams prefixless regex delimiters without waiting for the end of the stream", () => {
      const parser = new ResponseParser(qwen_serve_template, { prefix: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" });
      expect(parser.feed("<think>\nhmm\n</think>\n\n")).toEqual([
        { type: "region_open", field: "thinking" },
        { type: "region_chunk", field: "thinking", text: "\nhmm\n", dirty: false },
        { type: "region_close", field: "thinking", value: "hmm" },
      ]);
      // Trailing whitespace may still start a delimiter, so it is held back
      expect(parser.feed("The capital ")).toEqual([
        { type: "region_open", field: "content" },
        { type: "region_chunk", field: "content", text: "\n\nThe capital", dirty: false },
      ]);
      // A regex delimiter that ends at the buffer edge could still grow, so it closes on the next input
      expect(parser.feed("is Paris.<|im_end|>")).toEqual([{ type: "region_chunk", field: "content", text: " is Paris.", dirty: false }]);
      expect(parser.finalize()).toEqual([{ role: "assistant", thinking: "hmm", content: "The capital is Paris." }, [{ type: "region_close", field: "content", value: "The capital is Paris." }]]);
    });

    it("long regex open pattern streams character by character", () => {
      const [, template, text] = STREAMING_FIXTURES[3];
      const expected = parse_response(text, template, { prefix: "" });
      expect(expected.tool_calls[0].function.name).toEqual("get_current_weather");
      expect(stream_all(template, chunk_fixed(text, 1)).message).toEqual(expected);
    });

    it("feed after finalize raises", () => {
      const parser = new ResponseParser(smollm_template, { prefix: "" });
      parser.feed("<think>x</think>");
      parser.finalize();
      expect(() => parser.feed("more")).toThrow("already finalized");
      expect(() => parser.finalize()).toThrow("already finalized");
    });

    it("empty input streams cleanly", () => {
      const parser = new ResponseParser(smollm_template, { prefix: "" });
      expect(parser.feed("")).toEqual([]);
      expect(parser.finalize()).toEqual([{ role: "assistant" }, []]);
    });
  });

  describe("Prefix and truncation", () => {
    it("prefix lands inside an explicit region", () => {
      const prompt = "<|im_start|>system\nYou are helpful<|im_end|>\n" + "<|im_start|>user\nHi<|im_end|>\n" + "<|im_start|>assistant\n<think>\n";
      const parser = new ResponseParser(qwen3_template, { prefix: prompt });
      expect(parser.initial_events.map((e) => [e.type, e.field])).toEqual([
        ["region_open", "thinking"],
        ["region_chunk", "thinking"],
      ]);
      const events = parser.feed("Let me think...</think>");
      const [message] = parser.finalize();
      expect(message).toEqual({ role: "assistant", thinking: "Let me think..." });
      expect(events.map((e) => e.type)).toEqual(["region_chunk", "region_close"]);
      expect(events[1].field).toEqual("thinking");
    });

    it("prefix is truncated to the last anchor", () => {
      const prompt = "<|im_start|>system\nA<|im_end|>\n" + "<|im_start|>user\nB<|im_end|>\n" + "<|im_start|>assistant\nEarlier reply<|im_end|>\n" + "<|im_start|>user\nFollowup<|im_end|>\n" + "<|im_start|>assistant\n<think>\n";
      const parser = new ResponseParser(qwen3_template, { prefix: prompt });
      expect(parser.initial_events.filter((e) => e.type === "region_open").map((e) => e.field)).toEqual(["thinking"]);
      parser.feed("done</think>");
      expect(parser.finalize()[0]).toEqual({ role: "assistant", thinking: "done" });
    });

    it("rejects templates without an anchor at load time", () => {
      const { start_anchor, ...anchorless } = qwen3_template;
      expect(() => new ResponseParser(anchorless)).toThrow("start_anchor");
    });

    it("requires a prefix", () => {
      expect(() => new ResponseParser(qwen3_template)).toThrow("requires `prefix`");
      expect(() => parse_response("hi", qwen3_template)).toThrow("requires `prefix`");
    });

    it("falls back to the whole prefix when the anchor is not found", () => {
      const parser = new ResponseParser(qwen3_template, { prefix: "<think>\n" });
      expect(parser.initial_events.filter((e) => e.type === "region_open").map((e) => e.field)).toEqual(["thinking"]);
      parser.feed("hi</think>");
      expect(parser.finalize()[0]).toEqual({ role: "assistant", thinking: "hi" });
    });

    it("streaming with a prefix matches the one-shot parse", () => {
      const prompt = "<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nB<|im_end|>\n<|im_start|>assistant\n<think>\n";
      for (const [name, template, text] of STREAMING_FIXTURES) {
        if (name !== "qwen3" && name !== "smollm") continue;
        const via_prefix = parse_response(text, template, { prefix: prompt });
        for (const step of [1, 3, 7, 31]) {
          expect(stream_all(template, chunk_fixed(text, step), { prefix: prompt }).message).toEqual(via_prefix);
        }
      }
    });

    it("guards against history bleed through the prefix, not the response", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|im_start|>assistant\n",
        fields: { content: { close_pattern: String.raw`\Z`, content: "text" } },
      };
      const clean = { role: "assistant", content: "Hello there!" };
      expect(parse_response("Hello there!", template, { prefix: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" })).toEqual(clean);
      expect(parse_response("Hello there!", template, { prefix: "" })).toEqual(clean);
      // An anchor inside the response is treated as content, never as a history boundary.
      const gpt_oss_gen = "<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>answer";
      expect(parse_response(gpt_oss_gen, gpt_oss_template, { prefix: "" })).toEqual({ role: "assistant", thinking: "thinking", content: "answer" });
    });

    it("surfaces regions opened and closed inside the prefix", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "[BEGIN]",
        fields: {
          tag: { open: "<tag>", close: "</tag>", content: "text" },
          body: { close_pattern: "$", content: "text" },
        },
      };
      const parser = new ResponseParser(template, { prefix: "noise[BEGIN]<tag>silently consumed</tag>" });
      expect(parser.initial_events.map((e) => e.type)).toEqual(["region_open", "region_chunk", "region_close"]);
      expect(parser.initial_events.every((e) => e.field === "tag")).toBe(true);
      expect(parser.initial_events.at(-1).value).toEqual("silently consumed");
      parser.feed("real generated body");
      const [message] = parser.finalize();
      expect(message.tag).toEqual("silently consumed");
      expect(message.body).toEqual("real generated body");
    });

    it("prefix lands inside the implicit region", () => {
      const parser = new ResponseParser(smollm_template, { prefix: "<|im_start|>assistant\nSure, here is " });
      expect(parser.initial_events.filter((e) => e.type === "region_open").map((e) => e.field)).toEqual(["content"]);
      expect(parser.feed("the answer<|im_end|>").map((e) => e.type)).not.toContain("region_open");
    });

    it("prefix ending mid-delimiter", () => {
      const parser = new ResponseParser(qwen3_template, { prefix: "<|im_start|>assistant\n<thi" });
      expect(parser.initial_events).toEqual([]);
      expect(parser.feed("nk>real body</think>").map((e) => e.type)).toContain("region_open");
      expect(parser.finalize()[0]).toEqual({ role: "assistant", thinking: "real body" });
    });
  });

  describe("Tool argument coercion", () => {
    // xml-inline without a value_parser: parameter bodies stay raw strings until `tools` coerces them.
    const XML_STRING_ARGS_TEMPLATE = {
      defaults: { role: "assistant" },
      start_anchor: "<|im_start|>assistant\n",
      fields: {
        tool_calls: {
          open_pattern: String.raw`<tool_call>\s*<function=(?P<name>\w+)>`,
          close: "</tool_call>",
          repeats: true,
          content: "xml-inline",
          content_args: { tag_pattern: String.raw`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>` },
          transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
        },
      },
    };
    // kv-lines without a value_parser: values likewise stay raw strings for `tools` to cast.
    const KV_LINES_TOOLS_TEMPLATE = {
      defaults: { role: "assistant" },
      start_anchor: "<|im_start|>assistant\n",
      fields: {
        tool_calls: {
          open_pattern: String.raw`<tool_call>\s*<function=(?P<name>\w+)>\n`,
          close: "</tool_call>",
          repeats: true,
          content: "kv-lines",
          transform: { type: "function", function: { name: "{name}", arguments: "{content}" } },
        },
      },
    };
    const SET_ALARM_CALL = "<tool_call>\n<function=set_alarm>\n" + "<parameter=hour>\n7\n</parameter>\n" + "<parameter=enabled>\ntrue\n</parameter>\n" + "<parameter=label>\nwake up\n</parameter>\n" + "</function>\n</tool_call>";
    const set_alarm_tools = (properties) => [{ type: "function", function: { name: "set_alarm", parameters: { type: "object", properties } } }];
    const SET_ALARM_TOOLS = set_alarm_tools({ hour: { type: "integer" }, enabled: { type: "boolean" }, label: { type: "string" } });
    const first_tool_args = (message) => message.tool_calls[0].function.arguments;
    const parser_with_tools = (tools) => new ResponseParser(XML_STRING_ARGS_TEMPLATE, { prefix: "", tools });
    /** Coerce `raw` through a `label` parameter with the given JSON schema. */
    const coerce = (raw, schema) => {
      const model_out = `<tool_call>\n<function=set_alarm>\n<parameter=label>\n${raw}\n</parameter>\n</tool_call>`;
      return first_tool_args(parse_response(model_out, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: set_alarm_tools({ label: schema }) })).label;
    };

    it("casts declared types", () => {
      const tools = set_alarm_tools({
        count: { type: "integer" },
        ratio: { type: "number" },
        enabled: { type: "boolean" },
        tags: { type: "array" },
        note: { type: "string" },
      });
      const args = { count: "3", ratio: "1.5", enabled: "true", tags: '["a", "b"]', note: "hello", already_typed: 7, extra: "unscheduled" };
      const call = { type: "function", function: { name: "set_alarm", arguments: args } };
      expect(parser_with_tools(tools)._coerce_tool_calls(call)).toBe(call);
      expect(call.function.arguments).toEqual({ count: 3, ratio: 1.5, enabled: true, tags: ["a", "b"], note: "hello", already_typed: 7, extra: "unscheduled" });
    });

    it("follows Python's casting rules", () => {
      expect(coerce("not-a-number", { type: "integer" })).toEqual("not-a-number");
      expect(coerce("1_000", { type: "integer" })).toEqual(1000);
      expect(coerce("5", { type: ["integer", "null"] })).toEqual(5);
      expect(coerce("null", { type: ["integer", "null"] })).toBeNull();
      for (const [raw, expected] of [
        ["true", true],
        ["True", true],
        ["1", true],
        ["false", false],
        ["0", false],
      ]) {
        expect(coerce(raw, { type: "boolean" })).toEqual(expected);
      }
      // Non-boolean text stays a string rather than silently becoming false.
      expect(coerce("maybe", { type: "boolean" })).toEqual("maybe");
      // A JSON object body is only accepted for an `object` param, a JSON array only for `array`.
      expect(coerce("[1, 2]", { type: "object" })).toEqual("[1, 2]");
      expect(coerce('{"a": 1}', { type: "array" })).toEqual('{"a": 1}');
      expect(coerce('{"a": 1}', { type: "object" })).toEqual({ a: 1 });
      // NaN / inf are not valid JSON numbers, so a `number` param keeps the raw text.
      expect(coerce("NaN", { type: "number" })).toEqual("NaN");
    });

    it("handles the JSON schema type dialects", () => {
      expect(coerce("7", { type: ["integer", "string"] })).toEqual(7);
      expect(coerce("true", { anyOf: [{ type: "boolean" }, { type: "string" }] })).toEqual(true);
      expect(coerce("null", { oneOf: [{ type: "number" }, { type: "null" }] })).toBeNull();
      expect(coerce("null", { type: "integer", nullable: true })).toBeNull();
      // Undescribed parameters resolve to no candidate types, making coercion a no-op.
      expect(coerce("7", { description: "no type" })).toEqual("7");
    });

    it("handles single calls and lists of calls", () => {
      const parser = parser_with_tools(SET_ALARM_TOOLS);
      const call = { type: "function", function: { name: "set_alarm", arguments: { hour: "7" } } };
      expect(parser._coerce_tool_calls(call)).toBe(call);
      expect(call.function.arguments).toEqual({ hour: 7 });
      // A list of calls (as produced by `transform_each`) is coerced element-wise.
      const calls = [{ type: "function", function: { name: "set_alarm", arguments: { hour: "9" } } }];
      expect(parser._coerce_tool_calls(calls)[0].function.arguments).toEqual({ hour: 9 });
      // Non-tool-call values pass through untouched.
      expect(parser._coerce_tool_calls("hello")).toEqual("hello");
    });

    it("coerces xml-inline string arguments", () => {
      expect(first_tool_args(parse_response(SET_ALARM_CALL, XML_STRING_ARGS_TEMPLATE, { prefix: "" }))).toEqual({ hour: "7", enabled: "true", label: "wake up" });
      expect(first_tool_args(parse_response(SET_ALARM_CALL, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ hour: 7, enabled: true, label: "wake up" });
      const one_of = set_alarm_tools({ hour: { oneOf: [{ type: "integer" }, { type: "null" }] } });
      expect(first_tool_args(parse_response(SET_ALARM_CALL, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: one_of })).hour).toEqual(7);
    });

    it("coerces on region close while streaming", () => {
      const parser = new ResponseParser(XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: SET_ALARM_TOOLS });
      const closes = [...chunk_fixed(SET_ALARM_CALL, 8)].flatMap((chunk) => parser.feed(chunk).filter((e) => e.type === "region_close" && e.field === "tool_calls"));
      expect(closes.length).toEqual(1);
      expect(closes[0].value.function.arguments).toEqual({ hour: 7, enabled: true, label: "wake up" });
    });

    it("coerces strings left by a value parser, but never reworks typed values", () => {
      const model_out = "<tool_call>\n<function=set_alarm>\n" + "<parameter=hour>\n007\n</parameter>\n" + "<parameter=enabled>\ntrue\n</parameter>\n" + "<parameter=label>\nwake up\n</parameter>\n" + "</function>\n</tool_call>";
      expect(first_tool_args(parse_response(model_out, qwen3_template, { prefix: "" }))).toEqual({ hour: "007", enabled: true, label: "wake up" });
      expect(first_tool_args(parse_response(model_out, qwen3_template, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ hour: 7, enabled: true, label: "wake up" });

      const label_out = "<tool_call>\n<function=set_alarm>\n<parameter=label>\n1.50\n</parameter>\n</tool_call>";
      expect(first_tool_args(parse_response(label_out, qwen3_template, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ label: 1.5 });
      expect(first_tool_args(parse_response(label_out, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ label: "1.50" });
    });

    it("coerces kv-lines string arguments", () => {
      const model_out = "<tool_call>\n<function=set_alarm>\nhour: 7\nenabled: true\n</tool_call>";
      expect(first_tool_args(parse_response(model_out, KV_LINES_TOOLS_TEMPLATE, { prefix: "" }))).toEqual({ hour: "7", enabled: "true" });
      expect(first_tool_args(parse_response(model_out, KV_LINES_TOOLS_TEMPLATE, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ hour: 7, enabled: true });
    });

    it("leaves non-tool-call regions untouched", () => {
      const template = {
        defaults: { role: "assistant" },
        start_anchor: "<|im_start|>assistant\n",
        fields: {
          citation: {
            open_pattern: String.raw`<cite source=(?P<name>\w+)>`,
            close: "</cite>",
            content: "xml-inline",
            content_args: {
              tag_pattern: String.raw`<(?P<key>\w+)>\s*(?P<value>.*?)\s*</\1>`,
              value_parser: { name: "json", args: { allow_non_json: true } },
            },
            transform: { source: "{name}", fields: "{content}" },
          },
        },
      };
      const model_out = "<cite source=set_alarm><label>1.50</label><hour>7</hour></cite>";
      const expected = { source: "set_alarm", fields: { label: 1.5, hour: 7 } };
      expect(parse_response(model_out, template, { prefix: "" }).citation).toEqual(expected);
      expect(parse_response(model_out, template, { prefix: "", tools: SET_ALARM_TOOLS }).citation).toEqual(expected);
    });

    it("casts merged duplicate arguments element-wise", () => {
      const template = structuredClone(XML_STRING_ARGS_TEMPLATE);
      template.fields.tool_calls.content_args.merge_duplicates = true;
      const model_out = "<tool_call>\n<function=set_alarm>\n" + "<parameter=hour>\n7\n</parameter>\n" + "<parameter=hour>\n9\n</parameter>\n" + "</function>\n</tool_call>";
      expect(first_tool_args(parse_response(model_out, template, { prefix: "" }))).toEqual({ hour: ["7", "9"] });
      expect(first_tool_args(parse_response(model_out, template, { prefix: "", tools: SET_ALARM_TOOLS }))).toEqual({ hour: [7, 9] });
      // Elements that don't cast, and non-string elements, are left as they are.
      const call = { type: "function", function: { name: "set_alarm", arguments: { hour: ["7", "x", 9] } } };
      parser_with_tools(SET_ALARM_TOOLS)._coerce_tool_calls(call);
      expect(call.function.arguments).toEqual({ hour: [7, "x", 9] });
    });

    it("does not cast an already-decoded array element-wise", () => {
      const tools = set_alarm_tools({ groups: { type: "array", items: { type: "string" } } });
      const call = { type: "function", function: { name: "set_alarm", arguments: { groups: ["[1,2]", "[]"] } } };
      parser_with_tools(tools)._coerce_tool_calls(call);
      expect(call.function.arguments).toEqual({ groups: ["[1,2]", "[]"] });
    });

    it("ignores an unusable function name", () => {
      const call = { type: "function", function: { name: ["set_alarm"], arguments: { hour: "7" } } };
      expect(parser_with_tools(SET_ALARM_TOOLS)._coerce_tool_calls(call)).toEqual(call);
      expect(call.function.arguments).toEqual({ hour: "7" });
    });

    it("keeps scalar text for unions with a container type", () => {
      const union = set_alarm_tools({ label: { anyOf: [{ type: "string" }, { type: "object" }] } });
      const model_out = "<tool_call>\n<function=set_alarm>\n<parameter=label>\n1.50\n</parameter>\n</tool_call>";
      expect(first_tool_args(parse_response(model_out, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: union }))).toEqual({ label: "1.50" });
      const object_body = '<tool_call>\n<function=set_alarm>\n<parameter=label>\n{"a": 1}\n</parameter>\n</tool_call>';
      expect(first_tool_args(parse_response(object_body, XML_STRING_ARGS_TEMPLATE, { prefix: "", tools: union }))).toEqual({ label: { a: 1 } });
    });

    it("is a no-op for typed JSON tool calls", () => {
      const model_out = "<|START_THINKING|>x<|END_THINKING|>" + '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "set_alarm", ' + '"parameters": {"hour": 7, "enabled": true}}]<|END_ACTION|><|END_OF_TURN_TOKEN|>';
      const without = parse_response(model_out, cohere_template, { prefix: "" });
      const with_tools = parse_response(model_out, cohere_template, { prefix: "", tools: SET_ALARM_TOOLS });
      expect(with_tools).toEqual(without);
      expect(first_tool_args(with_tools)).toEqual({ hour: 7, enabled: true });
    });
  });

  describe("JavaScript port", () => {
    const template_with = (fields) => ({ defaults: { role: "assistant" }, start_anchor: "<a>", fields });

    it("does not share mutable defaults across parses", () => {
      const template = { ...template_with({ tool_calls: { open: "<tool>", close: "</tool>", repeats: true, content: "json" } }), defaults: { role: "assistant", tool_calls: [] } };
      const response = '<tool>{"name":"weather"}</tool>';
      expect(parse_response(response, template, { prefix: "" }).tool_calls).toEqual([{ name: "weather" }]);
      expect(parse_response(response, template, { prefix: "" }).tool_calls).toEqual([{ name: "weather" }]);
      expect(template.defaults.tool_calls).toEqual([]);
    });

    it("keeps model-generated keys away from object prototypes", () => {
      const parsed = parse_response("__proto__=safe", template_with({ content: { content: "kv-lines", content_args: { kv_sep: "=" } } }), { prefix: "" });
      expect(Object.getPrototypeOf(parsed.content)).toBe(Object.prototype);
      expect(Object.hasOwn(parsed.content, "__proto__")).toBe(true);
    });

    it("uses Python's content parser semantics", () => {
      const parse = (content, text) => parse_response(text, template_with({ content: { content } }), { prefix: "" }).content;
      expect(parse("bool", "yes")).toBe(false);
      expect(parse("int", " -1_000 ")).toBe(-1000);
      expect(parse("float", "1e3")).toBe(1000);
      expect(parse("float", "-inf")).toBe(-Infinity);
      expect(() => parse("int", "1.5")).toThrow("invalid literal for int()");
      // Only keys directly after `{` or `,` are quoted, so string contents are left alone
      const json = { t: { open: "<t>", close: "</t>", content: "json", content_args: { unquoted_keys: true } } };
      expect(parse_response('<t>{"text": "Note, warning: hot",n:1}</t>', template_with(json), { prefix: "" }).t).toEqual({ text: "Note, warning: hot", n: 1 });
    });

    it("translates Python regex syntax and semantics", () => {
      const tool = (open_pattern) => template_with({ t: { open_pattern, close: "</t>", transform: { name: "{name}", content: "{content}" } }, content: {} });
      // Python's `\w` is Unicode-aware
      expect(parse_response("<call:météo>x</t>", tool(String.raw`<call:(?P<name>\w+)>`), { prefix: "" }).t).toEqual({ name: "météo", content: "x" });
      // Escaped punctuation, lone braces and named backreferences
      expect(parse_response('{"x"}y</t>', tool(String.raw`(?P<name>\{\"x\"})(?P=name)?`), { prefix: "" }).t).toEqual({ name: '{"x"}', content: "y" });
      // Global inline flags
      expect(parse_response("<CALL:a>x</t>", tool(String.raw`(?i)<call:(?P<name>\w)>`), { prefix: "" }).t).toEqual({ name: "a", content: "x" });
      // Optional named groups that don't participate are not captured
      expect(() => parse_response("<t>x</t>", tool(String.raw`<t(?: (?P<name>\w+))?>`), { prefix: "" })).toThrow("'{name}' is not defined");
      // Without MULTILINE, `$` also matches before a trailing newline
      expect(parse_response("<c>body\n", template_with({ c: { open: "<c>", close_pattern: "$", content_args: { strip: false } } }), { prefix: "" })).toEqual({ role: "assistant", c: "body" });
    });

    it("reports unsupported regex syntax", () => {
      for (const open_pattern of ["(?>atomic)", "a++", "(?x)verbose", String.raw`\q`, "(unclosed"]) {
        expect(() => parse_response("", template_with({ t: { open_pattern, close: "</t>" } }), { prefix: "" })).toThrow("invalid open_pattern regex");
      }
    });
  });
});
