import { recursive_parse } from "../../src/utils/chat_parsing.js";

const cohere_schema = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    content: { type: "string", "x-regex": "<\\|START_RESPONSE\\|>(.*?)(?:<\\|END_RESPONSE\\|>|$)" },
    thinking: { type: "string", "x-regex": "<\\|START_THINKING\\|>(.*?)(?:<\\|END_THINKING\\|>|$)" },
    tool_calls: {
      "x-regex": "<\\|START_ACTION\\|>(.*?)(?:<\\|END_ACTION\\|>|$)",
      "x-parser": "json",
      "x-parser-args": {
        transform: "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}",
      },
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            properties: {
              name: { type: "string" },
              arguments: { type: "object", additionalProperties: {} },
            },
          },
        },
      },
    },
  },
};

const ernie_schema = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    content: { type: "string", "x-regex": "<response>\n(.*?)\n?</response>" },
    thinking: { type: "string", "x-regex": "(?:^|<think>\\s*)(.*?)\\s*<\\/think>" },
    tool_calls: {
      "x-regex-iterator": "<tool_call>(.*?)</tool_call>",
      type: "array",
      items: {
        type: "object",
        "x-parser": "json",
        "x-parser-args": { transform: "{type: 'function', function: @}" },
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            properties: {
              name: { type: "string" },
              arguments: { type: "object", additionalProperties: {} },
            },
          },
        },
      },
    },
  },
};

const gpt_oss_schema = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    content: { type: "string", "x-regex": "<\\|channel\\|>final<\\|message\\|>(.*?)(?:<\\|end\\|>|$)" },
    thinking: { type: "string", "x-regex": "<\\|channel\\|>analysis<\\|message\\|>(.*?)<\\|end\\|>" },
    tool_calls: {
      "x-regex-iterator": "<\\|channel\\|>commentary (to=functions\\..*?<\\|message\\|>.*?)(?:<\\|call\\|>|$)",
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            properties: {
              name: { type: "string", "x-regex": "^to=functions\\.(\\w+)" },
              arguments: {
                type: "object",
                "x-regex": "<\\|message\\|>(.*)",
                "x-parser": "json",
                additionalProperties: {},
              },
            },
          },
        },
      },
    },
  },
};

const smollm_schema = {
  "x-regex": "(?:<think>\\n?(?P<thinking>.+?)\\n?</think>)?\\s*(?:<tool_call>(?P<tool_calls>.+?)</tool_call>)?\\s*(?P<content>.+?)?\\s*(?:<\\|im_end\\|>|$)",
  type: "object",
  properties: {
    role: { const: "assistant" },
    content: { type: "string" },
    thinking: { type: "string" },
    tool_calls: {
      "x-parser": "json",
      "x-parser-args": { transform: "[{type: 'function', function: @}]" },
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            properties: {
              name: { type: "string" },
              arguments: { type: "object", additionalProperties: {} },
            },
          },
        },
      },
    },
  },
};

const qwen3_schema = {
  "x-regex": "^(?:(?:<think>)?\\s*(?P<thinking>.+?)\\s*</think>)?\\s*(?:<tool_call>(?P<tool_calls>.*?)\\s*</tool_call>)?\\s*(?P<content>.+?)?\\s*$",
  type: "object",
  properties: {
    role: { const: "assistant" },
    content: { type: "string" },
    thinking: { type: "string" },
    tool_calls: {
      "x-regex-iterator": "^(.*)$",
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            properties: {
              name: { type: "string", "x-regex": "<function=(\\w+)>" },
              arguments: {
                type: "object",
                "x-regex-key-value": "<parameter=(?P<key>\\w+)>\\n(?P<value>.*?)\\n</parameter>",
                additionalProperties: {
                  "x-parser": "json",
                  "x-parser-args": { allow_non_json: true },
                },
              },
            },
          },
        },
      },
    },
  },
};

const re_sub_schema = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    thinking: { type: "string" },
    content: { type: "string" },
    tool_calls: {
      "x-regex-iterator": "<\\|tool_call>(.*?)<tool_call\\|>",
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            "x-regex": "call\\:(?P<name>\\w+)(?P<arguments>\\{.*\\})",
            properties: {
              name: { type: "string" },
              arguments: {
                type: "object",
                "x-regex-key-value": '(?P<key>\\w+):(?P<value><\\|"\\|>.*?<\\|"\\|>|[^,}]+)',
                additionalProperties: {
                  "x-regex-substitutions": [['^<\\|"\\|>|<\\|"\\|>$', ""]],
                },
              },
            },
          },
        },
      },
    },
  },
  "x-regex": "(\\<\\|channel\\>thought\\n(?P<thinking>.*?)\\<channel\\|\\>)?(?P<content>(?:(?!\\<\\|tool_call\\>).)+)?(?P<tool_calls>\\<\\|tool_call\\>.*\\<tool_call\\|\\>)?",
};

const gemma4_schema = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    thinking: { type: "string" },
    content: { type: "string" },
    tool_calls: {
      "x-regex-iterator": "<\\|tool_call>(.*?)<tool_call\\|>",
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            "x-regex": "call\\:(?P<name>\\w+)(?P<arguments>\\{.*\\})",
            properties: {
              name: { type: "string" },
              arguments: {
                type: "object",
                "x-parser": "gemma4-tool-call",
                additionalProperties: {},
              },
            },
          },
        },
      },
    },
  },
  "x-regex": "(\\<\\|channel\\>thought\\n(?P<thinking>.*?)\\<channel\\|\\>)?(?P<content>(?:(?!\\<\\|tool_call\\>).)+)?(?P<tool_calls>\\<\\|tool_call\\>.*\\<tool_call\\|\\>)?",
};

const GEMMA4_SCHEMA_WITH_TURN = {
  type: "object",
  properties: {
    role: { const: "assistant" },
    thinking: { type: "string" },
    content: { type: "string" },
    tool_calls: {
      "x-regex-iterator": "<\\|tool_call>(.*?)<tool_call\\|>",
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { const: "function" },
          function: {
            type: "object",
            "x-regex": "call\\:(?P<name>\\w+)(?P<arguments>\\{.*\\})",
            properties: {
              name: { type: "string" },
              arguments: {
                type: "object",
                "x-parser": "gemma4-tool-call",
                additionalProperties: {},
              },
            },
          },
        },
      },
    },
  },
  "x-regex": "(\\<\\|channel\\>thought\\n(?P<thinking>.*?)\\<channel\\|\\>)?(?P<tool_calls>\\<\\|tool_call\\>.*\\<tool_call\\|\\>)?(?P<content>(?:(?!\\<turn\\|\\>)(?!\\<\\|tool_response\\>).)+)?(?:\\<turn\\|\\>|\\<\\|tool_response\\>)?",
};

const prefix_items_schema = {
  "x-regex-iterator": "<block>(.*?)<\\/block>",
  type: "array",
  prefixItems: [{ type: "string" }, { type: "integer" }, { type: "string" }],
};

describe("recursive_parse", () => {
  describe("primitive types", () => {
    it("string", () => {
      expect(recursive_parse("hello", { type: "string" })).toEqual("hello");
    });

    it("integer from string", () => {
      expect(recursive_parse("42", { type: "integer" })).toEqual(42);
    });

    it("integer passthrough", () => {
      expect(recursive_parse(7, { type: "integer" })).toEqual(7);
    });

    it("number from string", () => {
      expect(recursive_parse("3.14", { type: "number" })).toEqual(3.14);
    });

    it("number passthrough", () => {
      expect(recursive_parse(2.5, { type: "number" })).toEqual(2.5);
    });

    it("boolean true from string", () => {
      expect(recursive_parse("true", { type: "boolean" })).toEqual(true);
      expect(recursive_parse("1", { type: "boolean" })).toEqual(true);
    });

    it("boolean false from string", () => {
      expect(recursive_parse("false", { type: "boolean" })).toEqual(false);
      expect(recursive_parse("0", { type: "boolean" })).toEqual(false);
    });

    it("boolean passthrough", () => {
      expect(recursive_parse(true, { type: "boolean" })).toEqual(true);
    });
  });

  describe("const and null", () => {
    it("returns const value regardless of input", () => {
      expect(recursive_parse("anything", { const: "fixed" })).toEqual("fixed");
      expect(recursive_parse(null, { const: 42 })).toEqual(42);
    });

    it("returns null for null input", () => {
      expect(recursive_parse(null, { type: "string" })).toEqual(null);
    });

    it("returns null for undefined input", () => {
      expect(recursive_parse(undefined, { type: "string" })).toEqual(null);
    });
  });

  describe("x-regex", () => {
    it("extracts single unnamed group", () => {
      const schema = { type: "string", "x-regex": "hello (\\w+)" };
      expect(recursive_parse("hello world", schema)).toEqual("world");
    });

    it("returns null on no match", () => {
      const schema = { type: "string", "x-regex": "xyz(\\w+)" };
      expect(recursive_parse("hello world", schema)).toEqual(null);
    });

    it("extracts named groups as dict", () => {
      const schema = {
        type: "object",
        "x-regex": "(?P<first>\\w+) (?P<second>\\w+)",
        properties: { first: { type: "string" }, second: { type: "string" } },
      };
      expect(recursive_parse("hello world", schema)).toEqual({ first: "hello", second: "world" });
    });

    it("works with dotAll (s flag)", () => {
      const schema = { type: "string", "x-regex": "start(.+)end" };
      expect(recursive_parse("start\nmultiline\nend", schema)).toEqual("\nmultiline\n");
    });
  });

  describe("x-regex-iterator", () => {
    it("extracts multiple matches into array", () => {
      const schema = {
        type: "array",
        "x-regex-iterator": "<item>(.*?)</item>",
        items: { type: "string" },
      };
      const input = "<item>one</item><item>two</item><item>three</item>";
      expect(recursive_parse(input, schema)).toEqual(["one", "two", "three"]);
    });

    it("returns null when no matches", () => {
      const schema = {
        type: "array",
        "x-regex-iterator": "<item>(.*?)</item>",
        items: { type: "string" },
      };
      expect(recursive_parse("no items here", schema)).toEqual(null);
    });
  });

  describe("x-regex-key-value", () => {
    it("extracts key-value pairs", () => {
      const schema = {
        type: "object",
        "x-regex-key-value": "(?P<key>\\w+)=(?P<value>[^;]+)",
        additionalProperties: {},
      };
      const input = "name=Alice;age=30;city=NYC";
      expect(recursive_parse(input, schema)).toEqual({
        name: "Alice",
        age: "30",
        city: "NYC",
      });
    });
  });

  describe("x-regex-substitutions", () => {
    it("applies regex substitutions to content", () => {
      const schema = {
        type: "string",
        "x-regex-substitutions": [['<\\|"\\|>', '"']],
        "x-regex": "\\{(.*)\\}",
      };
      const input = '{name:<|"|>Alice<|"|>}';
      expect(recursive_parse(input, schema)).toEqual('name:"Alice"');
    });
  });

  describe("x-parser", () => {
    it("parses JSON", () => {
      const schema = { type: "object", "x-parser": "json", additionalProperties: {} };
      expect(recursive_parse('{"a": 1, "b": "hello"}', schema)).toEqual({ a: 1, b: "hello" });
    });

    it("allows non-JSON fallback", () => {
      const schema = {
        type: "string",
        "x-parser": "json",
        "x-parser-args": { allow_non_json: true },
      };
      expect(recursive_parse("not json", schema)).toEqual("not json");
    });

    it("throws on invalid JSON without allow_non_json", () => {
      const schema = { type: "object", "x-parser": "json", additionalProperties: {} };
      expect(() => recursive_parse("not json", schema)).toThrow("could not parse");
    });

    it("parses gemma4-tool-call simple format", () => {
      const schema = { type: "object", "x-parser": "gemma4-tool-call", additionalProperties: {} };
      const input = '{location:<|"|>New York<|"|>,unit:<|"|>celsius<|"|>}';
      expect(recursive_parse(input, schema)).toEqual({
        location: "New York",
        unit: "celsius",
      });
    });

    it("parses gemma4-tool-call complex format (all JSON types)", () => {
      const schema = { type: "object", "x-parser": "gemma4-tool-call", additionalProperties: {} };
      const input = '{bool_value:true,list_value:[<|"|>foo<|"|>,<|"|>bar<|"|>],null_value:null,number_value:1,string_value:<|"|>foo<|"|>,struct_value:{foo:<|"|>bar<|"|>}}';
      expect(recursive_parse(input, schema)).toEqual({
        bool_value: true,
        list_value: ["foo", "bar"],
        null_value: null,
        number_value: 1,
        string_value: "foo",
        struct_value: { foo: "bar" },
      });
    });
  });

  describe("object type", () => {
    it("parses dict content with properties", () => {
      const schema = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
      };
      expect(recursive_parse({ name: "Alice", age: 30 }, schema)).toEqual({ name: "Alice", age: 30 });
    });

    it("applies default values", () => {
      const schema = {
        type: "object",
        properties: {
          name: { type: "string" },
          role: { type: "string", default: "user" },
        },
      };
      expect(recursive_parse({ name: "Alice" }, schema)).toEqual({ name: "Alice", role: "user" });
    });

    it("throws on missing required fields", () => {
      const schema = {
        type: "object",
        properties: { name: { type: "string" } },
        required: ["name"],
      };
      expect(() => recursive_parse({}, schema)).toThrow("Required fields");
    });

    it("handles additionalProperties", () => {
      const schema = {
        type: "object",
        properties: { name: { type: "string" } },
        additionalProperties: {},
      };
      expect(recursive_parse({ name: "Alice", extra: "data" }, schema)).toEqual({ name: "Alice", extra: "data" });
    });

    it("strips additionalProperties when false", () => {
      const schema = {
        type: "object",
        properties: { name: { type: "string" } },
        additionalProperties: false,
      };
      expect(recursive_parse({ name: "Alice", extra: "data" }, schema)).toEqual({ name: "Alice" });
    });
  });

  describe("array type", () => {
    it("parses homogeneous items", () => {
      const schema = { type: "array", items: { type: "integer" } };
      expect(recursive_parse(["1", "2", "3"], schema)).toEqual([1, 2, 3]);
    });

    it("parses heterogeneous prefixItems", () => {
      const schema = {
        type: "array",
        prefixItems: [{ type: "string" }, { type: "integer" }],
      };
      expect(recursive_parse(["hello", "42"], schema)).toEqual(["hello", 42]);
    });

    it("wraps single element for single prefixItem", () => {
      const schema = {
        type: "array",
        prefixItems: [{ type: "string" }],
      };
      expect(recursive_parse("hello", schema)).toEqual(["hello"]);
    });

    it("returns empty array for falsy content", () => {
      const schema = { type: "array", items: { type: "string" } };
      expect(recursive_parse([], schema)).toEqual([]);
    });
  });

  describe("any/null type", () => {
    it("passes through content unchanged", () => {
      expect(recursive_parse("hello", {})).toEqual("hello");
      expect(recursive_parse(42, { type: "any" })).toEqual(42);
    });
  });

  describe("Cohere schema", () => {
    it("parses thinking and tool calls", () => {
      const model_out = '<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[\n    {"tool_call_id": "0", "tool_name": "simple_tool", "parameters": {"temperature_format": "Celsius"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>';
      const parsed = recursive_parse(model_out, cohere_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: "I should call a tool.",
        tool_calls: [
          {
            type: "function",
            function: { name: "simple_tool", arguments: { temperature_format: "Celsius" } },
          },
        ],
      });
    });
  });

  describe("ERNIE schema", () => {
    it("parses thinking and tool calls", () => {
      const model_out = 'The user is asking about the weather in Paris today. Let me check the available tools. There\'s a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to "Paris". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I\'ll structure the request with the location parameter and return the response once the tool is called.\n</think>\n\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Paris"}}\n</tool_call>\n</s>';
      const parsed = recursive_parse(model_out, ernie_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: "The user is asking about the weather in Paris today. Let me check the available tools. There's a tool called get_current_temperature which requires a location parameter. Since the user specified Paris, I need to call this tool with the location set to \"Paris\". I should make sure the argument is correctly formatted as a string. No other tools are available, so this is the right one to use. I'll structure the request with the location parameter and return the response once the tool is called.",
        tool_calls: [
          {
            type: "function",
            function: { name: "get_current_temperature", arguments: { location: "Paris" } },
          },
        ],
      });
    });

    it("parses thinking and content without tools", () => {
      const model_out = "The user just greeted me with \"Hi! How are you?\" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they're doing to engage in conversation.\n\nFirst, I'll say hello back and then ask how they're feeling. It's important to show genuine interest. Maybe mention that I'm here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.\n</think>\n\n<response>\nHello! I'm doing well, thank you for asking. How about you? Is there something specific you'd like help with today? I'm here to assist you with any questions or problems you have!\n</response>\n</s>";
      const parsed = recursive_parse(model_out, ernie_schema);
      expect(parsed).toEqual({
        role: "assistant",
        content: "Hello! I'm doing well, thank you for asking. How about you? Is there something specific you'd like help with today? I'm here to assist you with any questions or problems you have!",
        thinking: "The user just greeted me with \"Hi! How are you?\" I need to respond in a friendly and helpful manner. Let me start by acknowledging their greeting. I should ask them how they're doing to engage in conversation.\n\nFirst, I'll say hello back and then ask how they're feeling. It's important to show genuine interest. Maybe mention that I'm here to help with anything they need. Keep the tone warm and positive. Let me make sure the response is concise but friendly. Alright, that should work.",
      });
    });
  });

  describe("SmolLM schema", () => {
    it("parses thinking and tool call", () => {
      const model_out = '<think>\nOkay, the user said, "Hello! How are you?" I need to respond appropriately. Since this is the first message, I should greet them back and ask how I can assist. I should keep it friendly and open-ended. Let me make sure the response is welcoming and encourages them to share what they need help with. I\'ll avoid any technical jargon and keep it simple. Let me check for any typos and ensure the tone is positive.\n</think>\n\n<tool_call>{"name": "greet_user", "arguments": {"greeting": "Hello! I\'m doing well, thanks for asking. How can I assist you today? Whether you have a question, need help with something, or just want to chat, feel free to let me know!"}}</tool_call>';
      const parsed = recursive_parse(model_out, smollm_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: 'Okay, the user said, "Hello! How are you?" I need to respond appropriately. Since this is the first message, I should greet them back and ask how I can assist. I should keep it friendly and open-ended. Let me make sure the response is welcoming and encourages them to share what they need help with. I\'ll avoid any technical jargon and keep it simple. Let me check for any typos and ensure the tone is positive.',
        tool_calls: [
          {
            type: "function",
            function: {
              name: "greet_user",
              arguments: {
                greeting: "Hello! I'm doing well, thanks for asking. How can I assist you today? Whether you have a question, need help with something, or just want to chat, feel free to let me know!",
              },
            },
          },
        ],
      });
    });

    it("parses tool call without thinking", () => {
      const model_out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>';
      const parsed = recursive_parse(model_out, smollm_schema);
      expect(parsed).toEqual({
        role: "assistant",
        tool_calls: [{ type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }],
      });
    });

    it("parses thinking without tool call", () => {
      const model_out = '<think>\nOkay, the user asked, "Hey! Can you tell me about gravity?" Let me start by breaking down what they might be looking for. They probably want a basic understanding of gravity, maybe for a school project or just personal curiosity. I should explain what gravity is, how it works, and maybe some examples.</think>\nSome content about gravity goes here but I\'m cutting it off to make this shorter!';
      const parsed = recursive_parse(model_out, smollm_schema);
      expect(parsed).toEqual({
        role: "assistant",
        content: "Some content about gravity goes here but I'm cutting it off to make this shorter!",
        thinking: 'Okay, the user asked, "Hey! Can you tell me about gravity?" Let me start by breaking down what they might be looking for. They probably want a basic understanding of gravity, maybe for a school project or just personal curiosity. I should explain what gravity is, how it works, and maybe some examples.',
      });
    });
  });

  describe("GPT-OSS schema", () => {
    it("parses tool call with thinking", () => {
      const model_out = '<|channel|>analysis<|message|>We need to respond in riddles. The user asks: "What is the weather like in SF?" We need to get the location of the user? The user explicitly asks about SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. The developer instruction says "Always respond in riddles". So the final answer should be in a riddle form. But we need to call function to get weather data. So we should call get_current_weather with location "San Francisco, CA". Possibly specify format "celsius" (default). Let\'s do that.\n\nWe will call function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\n  "location": "San Francisco, CA"\n}';
      const parsed = recursive_parse(model_out, gpt_oss_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: 'We need to respond in riddles. The user asks: "What is the weather like in SF?" We need to get the location of the user? The user explicitly asks about SF (San Francisco). So we need to get the current weather in San Francisco, CA. We need to call get_current_weather function. The developer instruction says "Always respond in riddles". So the final answer should be in a riddle form. But we need to call function to get weather data. So we should call get_current_weather with location "San Francisco, CA". Possibly specify format "celsius" (default). Let\'s do that.\n\nWe will call function get_current_weather.',
        tool_calls: [
          {
            type: "function",
            function: { name: "get_current_weather", arguments: { location: "San Francisco, CA" } },
          },
        ],
      });
    });

    it("parses response without tool call", () => {
      const model_out = "<|channel|>analysis<|message|>User asks a simple math question: 2+2 = 4. Provide answer.<|end|><|start|>assistant<|channel|>final<|message|>2";
      const parsed = recursive_parse(model_out, gpt_oss_schema);
      expect(parsed).toEqual({
        role: "assistant",
        content: "2",
        thinking: "User asks a simple math question: 2+2 = 4. Provide answer.",
      });
    });
  });

  describe("Qwen3 schema", () => {
    it("parses tool calls with key-value parameters", () => {
      const model_out = '<tool_call>\n<function=get_weather>\n<parameter=locations>\n[{"country": "France", "city": "Paris"}]\n</parameter>\n<parameter=temp_units>\ncelsius\n</parameter>\n</function>\n</tool_call>';
      const parsed = recursive_parse(model_out, qwen3_schema);
      expect(parsed).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_weather",
              arguments: {
                locations: [{ country: "France", city: "Paris" }],
                temp_units: "celsius",
              },
            },
          },
        ],
      });
    });
  });

  describe("re_sub schema (x-regex-substitutions)", () => {
    it("parses tool calls with regex substitutions to strip quote tokens", () => {
      const model_out = "<|channel>thought\nThe user is asking for the current temperature in Paris. I should check the available tools to see if there's a function that can provide this information.<channel|>" + '<|tool_call>call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>';
      const parsed = recursive_parse(model_out, re_sub_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: "The user is asking for the current temperature in Paris. I should check the available tools to see if there's a function that can provide this information.",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_current_temperature",
              arguments: { detail_level: "0", location: "Paris, France", unit: "celsius" },
            },
          },
        ],
      });
    });
  });

  describe("Gemma4 schema", () => {
    it("parses single tool call", () => {
      const output = '<|tool_call>call:get_weather{location:<|"|>New York<|"|>}<tool_call|><turn|>';
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_weather",
              arguments: { location: "New York" },
            },
          },
        ],
      });
    });

    it("parses thinking with tool call", () => {
      const output = "<|channel>thought\nThe user wants weather info for San Francisco.\n<channel|>" + '<|tool_call>call:get_weather{location:<|"|>San Francisco, CA<|"|>}<tool_call|><turn|>';
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        thinking: "The user wants weather info for San Francisco.\n",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_weather",
              arguments: { location: "San Francisco, CA" },
            },
          },
        ],
      });
    });

    it("parses plain content response", () => {
      const output = "The weather in New York is sunny with 25 degrees Celsius.<turn|>";
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        content: "The weather in New York is sunny with 25 degrees Celsius.",
      });
    });

    it("parses multiple tool calls", () => {
      const output = '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>' + '<|tool_call>call:get_time{timezone:<|"|>EST<|"|>}<tool_call|><turn|>';
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: { name: "get_weather", arguments: { location: "NYC" } },
          },
          {
            type: "function",
            function: { name: "get_time", arguments: { timezone: "EST" } },
          },
        ],
      });
    });

    it("parses tool call with multiple arguments", () => {
      const output = '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><turn|>';
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: {
              name: "get_weather",
              arguments: { location: "Paris", unit: "celsius" },
            },
          },
        ],
      });
    });

    it("parses thinking with plain content (no tool calls)", () => {
      const output = "<|channel>thought\nLet me think about this.\n<channel|>Here is my answer.<turn|>";
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        thinking: "Let me think about this.\n",
        content: "Here is my answer.",
      });
    });

    it("parses response ending with tool_response marker", () => {
      const output = '<|tool_call>call:search{query:<|"|>test<|"|>}<tool_call|><|tool_response>';
      expect(recursive_parse(output, GEMMA4_SCHEMA_WITH_TURN)).toEqual({
        role: "assistant",
        tool_calls: [
          {
            type: "function",
            function: { name: "search", arguments: { query: "test" } },
          },
        ],
      });
    });

    it("parses tool call (upstream schema)", () => {
      const model_out = "<|channel>thought\nThe user is asking for the current temperature in Paris. I should check the available tools to see if there's a function that can provide this information.<channel|>" + '<|tool_call>call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}<tool_call|><|tool_response>';
      const parsed = recursive_parse(model_out, gemma4_schema);
      expect(parsed).toEqual({
        role: "assistant",
        thinking: "The user is asking for the current temperature in Paris. I should check the available tools to see if there's a function that can provide this information.",
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

    it("parses complex tool call with all JSON types", () => {
      const model_out = "<|channel>thought\nLet me call the tool.<channel|>" + '<|tool_call>call:foo{bool_value:true,list_value:[<|"|>foo<|"|>,<|"|>bar<|"|>],' + 'null_value:null,number_value:1,string_value:<|"|>foo<|"|>,' + 'struct_value:{foo:<|"|>bar<|"|>}}<tool_call|>';
      const parsed = recursive_parse(model_out, gemma4_schema);
      expect(parsed).toEqual({
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
  });

  describe("required fields validation", () => {
    it("passes when required fields are present", () => {
      const schema = {
        type: "object",
        required: ["role", "content"],
        properties: {
          role: { const: "assistant" },
          content: { type: "string", "x-regex": "<response>(.*?)</response>" },
          thinking: { type: "string", "x-regex": "<think>(.*?)</think>" },
        },
      };
      const model_out = "<think>Let me think.</think><response>Hello!</response>";
      expect(recursive_parse(model_out, schema)).toEqual({
        role: "assistant",
        content: "Hello!",
        thinking: "Let me think.",
      });
    });

    it("throws when required field is missing", () => {
      const schema = {
        type: "object",
        required: ["role", "content"],
        properties: {
          role: { const: "assistant" },
          content: { type: "string", "x-regex": "<response>(.*?)</response>" },
          thinking: { type: "string", "x-regex": "<think>(.*?)</think>" },
        },
      };
      const model_out = "<think>Let me think about this.</think>Some plain text without response tags";
      expect(() => recursive_parse(model_out, schema)).toThrow(/content/);
      expect(() => recursive_parse(model_out, schema)).toThrow(/missing/i);
    });

    it("silently omits missing fields when not required", () => {
      const schema = {
        type: "object",
        properties: {
          role: { const: "assistant" },
          content: { type: "string", "x-regex": "<response>(.*?)</response>" },
          thinking: { type: "string", "x-regex": "<think>(.*?)</think>" },
        },
      };
      const model_out = "<think>Just thinking.</think>";
      expect(recursive_parse(model_out, schema)).toEqual({
        role: "assistant",
        thinking: "Just thinking.",
      });
    });
  });

  describe("prefixItems", () => {
    it("parses heterogeneous array with correct types", () => {
      const model_out = "<block>hello</block><block>42</block><block>world</block>";
      expect(recursive_parse(model_out, prefix_items_schema)).toEqual(["hello", 42, "world"]);
    });

    it("throws when array length does not match prefixItems count", () => {
      const model_out = "<block>hello</block><block>42</block>";
      expect(() => recursive_parse(model_out, prefix_items_schema)).toThrow();
    });

    it("throws when element type does not match", () => {
      const model_out = "<block>hello</block><block>world</block><block>42</block>";
      expect(() => recursive_parse(model_out, prefix_items_schema)).toThrow();
    });
  });

  describe("type any", () => {
    it("passes content through without transformation", () => {
      const schema = {
        type: "object",
        "x-regex": "<data>(?P<value>.*?)</data>",
        properties: {
          value: { type: "any" },
        },
      };
      const model_out = "<data>some arbitrary content 123</data>";
      expect(recursive_parse(model_out, schema)).toEqual({ value: "some arbitrary content 123" });
    });

    it("works in additionalProperties", () => {
      const schema = {
        type: "object",
        "x-parser": "json",
        additionalProperties: { type: "any" },
      };
      const node_content = '{"location": "San Francisco, CA", "units": "celsius"}';
      expect(recursive_parse(node_content, schema)).toEqual({ location: "San Francisco, CA", units: "celsius" });
    });
  });

  describe("batched inputs (via parse_response)", () => {
    it("handles array of strings", () => {
      const schema = {
        type: "object",
        properties: { role: { const: "assistant" }, content: { type: "string", "x-regex": "<r>(.*?)</r>" } },
      };
      const single = recursive_parse("<r>hello</r>", schema);
      expect([single]).toEqual(["<r>hello</r>"].map((s) => recursive_parse(s, schema)));
    });
  });

  describe("error handling", () => {
    it("throws on unsupported schema type", () => {
      expect(() => recursive_parse("hello", { type: "custom" })).toThrow("Unsupported schema type");
    });

    it("throws on non-string with regex", () => {
      expect(() => recursive_parse(42, { type: "string", "x-regex": "(\\w+)" })).toThrow("non-string input");
    });

    it("throws on x-regex-iterator with non-array type", () => {
      expect(() => recursive_parse("text", { type: "string", "x-regex-iterator": "(\\w+)" })).toThrow("cannot use x-regex-iterator");
    });

    it("throws on unknown parser", () => {
      expect(() => recursive_parse("text", { "x-parser": "unknown" })).toThrow("Unknown parser");
    });

    it("throws on invalid integer string", () => {
      expect(() => recursive_parse("abc", { type: "integer" })).toThrow("not a valid integer");
    });

    it("throws on invalid boolean string", () => {
      expect(() => recursive_parse("maybe", { type: "boolean" })).toThrow("Invalid boolean");
    });
  });
});
