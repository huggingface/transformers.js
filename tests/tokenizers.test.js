import { AutoTokenizer } from "../src/tokenizers.js";
import { MAX_TOKENIZER_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "./init.js";
import { compare, collect_tests } from "./test_utils.js";

const TOKENIZER_TESTS = await collect_tests("tokenization");
describe("Tokenizers (model-specific)", () => {
  for (const [tokenizer_name, { TOKENIZER_CLASS, TEST_CONFIG, CUSTOM_TESTS }] of TOKENIZER_TESTS) {
    describe(tokenizer_name, () => {
      for (const model_id in TEST_CONFIG) {
        describe(model_id, () => {
          /** @type {import('../src/tokenizers.js').PreTrainedTokenizer} */
          let tokenizer;
          beforeAll(async () => {
            tokenizer = await TOKENIZER_CLASS.from_pretrained(model_id);
          }, MAX_TOKENIZER_LOAD_TIME);

          for (const [test_name, test_case] of Object.entries(TEST_CONFIG[model_id])) {
            test(test_name, () => {
              if (test_case.ids) {
                const ids = tokenizer.encode(test_case.text, {
                  text_pair: test_case.text_pair,
                });
                expect(ids).toEqual(test_case.ids);
              }
              if (test_case.tokens) {
                const tokens = tokenizer.tokenize(test_case.text, {
                  pair: test_case.text_pair,
                });
                expect(tokens).toEqual(test_case.tokens);
              }
              if (test_case.decoded) {
                const decoded = tokenizer.decode(test_case.ids);
                expect(decoded).toEqual(test_case.decoded);
              }
            });
          }
        });
      }
      // Run custom tests, if they exist
      CUSTOM_TESTS && describe("custom", CUSTOM_TESTS);
    });
  }
});

describe("Tokenizer padding/truncation", () => {
  const inputs = ["a", "b c"];
  const text_pair = ["d e", "f g h"];

  it("should create a jagged array", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

    {
      // support jagged array if `return_tensor=false`
      const output = tokenizer(inputs, {
        return_tensor: false,
      });
      const expected = {
        input_ids: [
          [101, 1037, 102],
          [101, 1038, 1039, 102],
        ],
        attention_mask: [
          [1, 1, 1],
          [1, 1, 1, 1],
        ],
        token_type_ids: [
          [0, 0, 0],
          [0, 0, 0, 0],
        ],
      };
      compare(output, expected);
    }

    {
      const output = tokenizer(inputs, {
        return_tensor: false,
        truncation: true,
        add_special_tokens: false,
      });
      const expected = {
        input_ids: [[1037], [1038, 1039]],
        attention_mask: [[1], [1, 1]],
        token_type_ids: [[0], [0, 0]],
      };
      compare(output, expected);
    }
  });

  it(
    "should create a tensor",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      {
        // Expected to throw error if jagged array
        expect(() => tokenizer(inputs)).toThrow("Unable to create tensor");
      }

      {
        // Truncation
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          truncation: true,
          max_length: 1,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([[1037n], [1038n]]);
        expect(attention_mask.tolist()).toEqual([[1n], [1n]]);
        expect(token_type_ids.tolist()).toEqual([[0n], [0n]]);
      }
      {
        // Truncation w/ text pair
        // TODO
      }

      {
        // Padding
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          padding: true,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 0n],
          [1038n, 1039n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 0n],
          [1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 0n],
          [0n, 0n],
        ]);
      }
      {
        // Padding w/ text pair
        const { input_ids, attention_mask, token_type_ids } = tokenizer(inputs, {
          text_pair,
          padding: true,
          add_special_tokens: false,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 1040n, 1041n, 0n, 0n],
          [1038n, 1039n, 1042n, 1043n, 1044n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 1n, 1n, 0n, 0n],
          [1n, 1n, 1n, 1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 1n, 1n, 0n, 0n],
          [0n, 0n, 1n, 1n, 1n],
        ]);
      }

      {
        // Truncation + padding
        const { input_ids, attention_mask, token_type_ids } = tokenizer(["a", "b c", "d e f"], {
          padding: true,
          truncation: true,
          add_special_tokens: false,
          max_length: 2,
        });

        expect(input_ids.tolist()).toEqual([
          [1037n, 0n],
          [1038n, 1039n],
          [1040n, 1041n],
        ]);
        expect(attention_mask.tolist()).toEqual([
          [1n, 0n],
          [1n, 1n],
          [1n, 1n],
        ]);
        expect(token_type_ids.tolist()).toEqual([
          [0n, 0n],
          [0n, 0n],
          [0n, 0n],
        ]);
      }
    },
    MAX_TEST_EXECUTION_TIME,
  );
});

describe("Token type ids", () => {
  it(
    "should correctly add token type ids",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      const model_inputs = tokenizer(["a b c", "d"], {
        text_pair: ["e f", "g h"],
        padding: true,
        truncation: true,
        return_tensor: false,
      });

      const expected = {
        input_ids: [
          [101, 1037, 1038, 1039, 102, 1041, 1042, 102],
          [101, 1040, 102, 1043, 1044, 102, 0, 0],
        ],
        token_type_ids: [
          [0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 1, 1, 1, 0, 0],
        ],
        attention_mask: [
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 0, 0],
        ],
      };

      compare(model_inputs, expected);
    },
    MAX_TEST_EXECUTION_TIME,
  );

  it(
    "should add token type ids if user requests them",
    async () => {
      const tokenizer = await AutoTokenizer.from_pretrained("Xenova/llama3-tokenizer-new");

      {
        // Without text pair
        const model_inputs = tokenizer("hello", {
          return_tensor: false,
          return_token_type_ids: true,
        });
        const expected = {
          input_ids: [128000, 15339],
          attention_mask: [1, 1],
          token_type_ids: [0, 0],
        };
        compare(model_inputs, expected);
      }

      {
        // With text pair
        const model_inputs = tokenizer("hello", {
          text_pair: "world",
          return_tensor: false,
          return_token_type_ids: true,
        });
        const expected = {
          input_ids: [128000, 15339, 128000, 14957],
          attention_mask: [1, 1, 1, 1],
          token_type_ids: [0, 0, 1, 1],
        };
        compare(model_inputs, expected);
      }
    },
    MAX_TEST_EXECUTION_TIME,
  );
});

describe("Edge cases", () => {
  it(
    "should not crash when encoding a very long string",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/t5-small");

      let text = String.prototype.repeat.call("Hello world! ", 50000);
      let encoded = tokenizer(text);
      expect(encoded.input_ids.data.length).toBeGreaterThan(100000);
    },
    MAX_TEST_EXECUTION_TIME,
  );

  it("should not take too long", async () => {
    let tokenizer = await AutoTokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2");

    let text = String.prototype.repeat.call("a", 50000);
    let token_ids = tokenizer.encode(text);
    compare(token_ids, [101, 100, 102]);
  }, 5000); // NOTE: 5 seconds

  it(
    "Special/added tokens with earlier partial matches",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/gemini-nano");
      {
        let token_ids = tokenizer.encode("\n", { add_special_tokens: false });
        compare(token_ids, [108]);
      }
      {
        let token_ids = tokenizer.encode("\n\n", { add_special_tokens: false });
        compare(token_ids, [109]); // Should not be [108, 108]
      }
    },
    MAX_TEST_EXECUTION_TIME,
  );

  it("many added tokens", async () => {
    let tokenizer = await AutoTokenizer.from_pretrained("onnx-community/orpheus-3b-0.1-ft-ONNX");

    let text = "hello world!";
    let token_ids = tokenizer.encode(text);
    compare(token_ids, [128000, 15339, 1917, 0]);
  }, 5000); // NOTE: 5 seconds
});

describe("Memory leak tests", () => {
  it("should not leak memory", async () => {
    const id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    const tokenizer = await AutoTokenizer.from_pretrained(id);

    // NB: Force garbage collection before measuring memory usage
    global.gc();
    const startMemory = process.memoryUsage().heapUsed;
    for (let i = 0; i < 2500; ++i) {
      const s = `${i} ${i} `.repeat(i);
      tokenizer.encode(s);
    }
    global.gc();
    const endMemory = process.memoryUsage().heapUsed;
    const memoryUsed = endMemory - startMemory;
    const memoryLimit = 1 * 1024 * 1024; // 1 MB
    expect(memoryUsed).toBeLessThan(memoryLimit);
  }, 30000); // Increase timeout to accommodate the memory leak test
});

describe("Extra decoding tests", () => {
  it(
    "should be able to decode the output of encode",
    async () => {
      let tokenizer = await AutoTokenizer.from_pretrained("Xenova/bert-base-uncased");

      let text = "hello world!";

      // Ensure all the following outputs are the same:
      // 1. Tensor of ids: allow decoding of 1D or 2D tensors.
      let encodedTensor = tokenizer(text);
      let decoded1 = tokenizer.decode(encodedTensor.input_ids, { skip_special_tokens: true });
      let decoded2 = tokenizer.batch_decode(encodedTensor.input_ids, { skip_special_tokens: true })[0];
      expect(decoded1).toEqual(text);
      expect(decoded2).toEqual(text);

      // 2. List of ids
      let encodedList = tokenizer(text, { return_tensor: false });
      let decoded3 = tokenizer.decode(encodedList.input_ids, { skip_special_tokens: true });
      let decoded4 = tokenizer.batch_decode([encodedList.input_ids], { skip_special_tokens: true })[0];
      expect(decoded3).toEqual(text);
      expect(decoded4).toEqual(text);
    },
    MAX_TEST_EXECUTION_TIME,
  );
});

describe("Chat templates", () => {
  it("should generate a chat template", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/mistral-tokenizer-v1");

    const chat = [
      { role: "user", content: "Hello, how are you?" },
      { role: "assistant", content: "I'm doing great. How can I help you today?" },
      { role: "user", content: "I'd like to show off how chat templating works!" },
    ];

    const text = tokenizer.apply_chat_template(chat, { tokenize: false });

    expect(text).toEqual("<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]");

    const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
    compare(input_ids, [1, 733, 16289, 28793, 22557, 28725, 910, 460, 368, 28804, 733, 28748, 16289, 28793, 28737, 28742, 28719, 2548, 1598, 28723, 1602, 541, 315, 1316, 368, 3154, 28804, 2, 28705, 733, 16289, 28793, 315, 28742, 28715, 737, 298, 1347, 805, 910, 10706, 5752, 1077, 3791, 28808, 733, 28748, 16289, 28793]);
  });

  it("should support multiple chat templates", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/c4ai-command-r-v01-tokenizer");

    // define conversation input:
    const conversation = [{ role: "user", content: "Whats the biggest penguin in the world?" }];
    // define documents to ground on:
    const documents = [
      { title: "Tall penguins", text: "Emperor penguins are the tallest growing up to 122 cm in height." },
      { title: "Penguin habitats", text: "Emperor penguins only live in Antarctica." },
    ];

    // render the RAG prompt as a string:
    const grounded_generation_prompt = tokenizer.apply_chat_template(conversation, {
      chat_template: "rag",
      tokenize: false,
      add_generation_prompt: true,

      documents,
      citation_mode: "accurate", // or "fast"
    });
    expect(grounded_generation_prompt).toEqual("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble\nThe instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.\n\n" + "# System Preamble\n## Basic Rules\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.\n\n" + "# User Preamble\n## Task and Context\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.\n\n## Style Guide\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>\nDocument: 0\ntitle: Tall penguins\ntext: Emperor penguins are the tallest growing up to 122 cm in height.\n\nDocument: 1\ntitle: Penguin habitats\ntext: Emperor penguins only live in Antarctica.\n</results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.\nFirstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.\nSecondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.\nThirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\nFinally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|>" + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
  });

  it("should support automatic chat template detection based on inputs", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/c4ai-command-r-plus-08-2024-tokenizer");

    // Examples adapted from https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024

    {
      // - default
      // define conversation input:
      const messages = [{ role: "user", content: "Hello, how are you?" }];

      // Format message with the command-r-plus-08-2024 chat template
      const prompt = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
      expect(prompt).toEqual("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
    }

    {
      // - tool_use
      // define conversation input:
      const conversation = [{ role: "user", content: "Whats the biggest penguin in the world?" }];

      // Define tools available for the model to use:
      const tools = [
        {
          name: "internet_search",
          description: "Returns a list of relevant document snippets for a textual query retrieved from the internet",
          parameter_definitions: {
            query: {
              description: "Query to search the internet with",
              type: "str",
              required: true,
            },
          },
        },
        {
          name: "directly_answer",
          description: "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
          parameter_definitions: {},
        },
      ];

      // render the tool use prompt as a string:
      const prompt = tokenizer.apply_chat_template(conversation, { tools, tokenize: false, add_generation_prompt: true });
      expect(prompt).toEqual('<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble\nThe instructions in this section override those in the task description and style guide sections. Don\'t answer questions that are harmful or immoral.\n\n# System Preamble\n## Basic Rules\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\'s requests, you cite your sources in your answers, according to those instructions.\n\n# User Preamble\n## Task and Context\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\'s needs as best you can, which will be wide-ranging.\n\n## Style Guide\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n```python\ndef internet_search(query: str) -> List[Dict]:\n    """Returns a list of relevant document snippets for a textual query retrieved from the internet\n\n    Args:\n        query (str): Query to search the internet with\n    """\n    pass\n```\n\n```python\ndef directly_answer() -> List[Dict]:\n    """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history\n    """\n    pass\n```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \'Action:\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>');
    }
  });

  it("should support user-defined chat template", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/llama-tokenizer");

    const chat = [
      { role: "user", content: "Hello, how are you?" },
      { role: "assistant", content: "I'm doing great. How can I help you today?" },
      { role: "user", content: "I'd like to show off how chat templating works!" },
    ];

    // https://discuss.huggingface.co/t/issue-with-llama-2-chat-template-and-out-of-date-documentation/61645/3
    const chat_template = (
      "{% if messages[0]['role'] == 'system' %}" +
      "{% set loop_messages = messages[1:] %}" + // Extract system message if it's present
      "{% set system_message = messages[0]['content'] %}" +
      "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}" +
      "{% set loop_messages = messages %}" + // Or use the default system message if the flag is set
      "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}" +
      "{% else %}" +
      "{% set loop_messages = messages %}" +
      "{% set system_message = false %}" +
      "{% endif %}" +
      "{% if loop_messages|length == 0 and system_message %}" + // Special handling when only sys message present
      "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}" +
      "{% endif %}" +
      "{% for message in loop_messages %}" + // Loop over all non-system messages
      "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}" +
      "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}" +
      "{% endif %}" +
      "{% if loop.index0 == 0 and system_message != false %}" + // Embed system message in first message
      "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}" +
      "{% else %}" +
      "{% set content = message['content'] %}" +
      "{% endif %}" +
      "{% if message['role'] == 'user' %}" + // After all of that, handle messages/roles in a fairly normal way
      "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}" +
      "{% elif message['role'] == 'system' %}" +
      "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}" +
      "{% elif message['role'] == 'assistant' %}" +
      "{{ ' '  + content.strip() + ' ' + eos_token }}" +
      "{% endif %}" +
      "{% endfor %}"
    )
      .replaceAll("USE_DEFAULT_PROMPT", true)
      .replaceAll("DEFAULT_SYSTEM_MESSAGE", "You are a helpful, respectful and honest assistant.");

    const text = tokenizer.apply_chat_template(chat, { tokenize: false, return_tensor: false, chat_template });

    expect(text).toEqual("<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]");

    // TODO: Add test for token_ids once bug in transformers is fixed.
  });

  it("should throw an error when no chat template is detected", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");

    const chat = [{ role: "user", content: "Hello, how are you?" }];

    expect(() => tokenizer.apply_chat_template(chat, { tokenize: false })).toThrow("tokenizer.chat_template is not set and no template argument was passed");
  });

  it("should support default parameters", async () => {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/Meta-Llama-3.1-Tokenizer");

    // Example adapted from https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct#tool-use-with-transformers
    const chat = [
      { role: "system", content: "You are a bot that responds to weather queries." },
      { role: "user", content: "Hey, what's the temperature in Paris right now?" },
    ];
    const tools = [{ type: "function", function: { name: "get_current_temperature", description: "Get the current temperature at a location.", parameters: { type: "object", properties: { location: { type: "string", description: 'The location to get the temperature for, in the format "City, Country"' } }, required: ["location"] }, return: { type: "number", description: "The current temperature at the specified location in the specified units, as a float." } } }];

    {
      // `tools` unset (will default to `null`)
      const text = tokenizer.apply_chat_template(chat, { tokenize: false });
      expect(text).toEqual("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a bot that responds to weather queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHey, what's the temperature in Paris right now?<|eot_id|>");

      const input_ids = tokenizer.apply_chat_template(chat, { tokenize: true, return_tensor: false });
      compare(input_ids, [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11164, 430, 31680, 311, 9282, 20126, 13, 128009, 128006, 882, 128007, 271, 19182, 11, 1148, 596, 279, 9499, 304, 12366, 1314, 1457, 30, 128009]);
    }

    {
      // `tools` set
      const text = tokenizer.apply_chat_template(chat, { tools, tokenize: false });
      expect(text).toEqual('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a bot that responds to weather queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.\n\n{\n    "type": "function",\n    "function": {\n        "name": "get_current_temperature",\n        "description": "Get the current temperature at a location.",\n        "parameters": {\n            "type": "object",\n            "properties": {\n                "location": {\n                    "type": "string",\n                    "description": "The location to get the temperature for, in the format \\"City, Country\\""\n                }\n            },\n            "required": [\n                "location"\n            ]\n        },\n        "return": {\n            "type": "number",\n            "description": "The current temperature at the specified location in the specified units, as a float."\n        }\n    }\n}\n\nHey, what\'s the temperature in Paris right now?<|eot_id|>');

      const input_ids = tokenizer.apply_chat_template(chat, { tools, tokenize: true, return_tensor: false });
      compare(input_ids, [128000, 128006, 9125, 128007, 271, 13013, 25, 6125, 27993, 198, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11164, 430, 31680, 311, 9282, 20126, 13, 128009, 128006, 882, 128007, 271, 22818, 279, 2768, 5865, 11, 4587, 6013, 449, 264, 4823, 369, 264, 734, 1650, 449, 1202, 6300, 6105, 430, 1888, 11503, 279, 2728, 10137, 382, 66454, 304, 279, 3645, 5324, 609, 794, 734, 836, 11, 330, 14105, 794, 11240, 315, 5811, 836, 323, 1202, 907, 7966, 5519, 539, 1005, 7482, 382, 517, 262, 330, 1337, 794, 330, 1723, 761, 262, 330, 1723, 794, 341, 286, 330, 609, 794, 330, 456, 11327, 54625, 761, 286, 330, 4789, 794, 330, 1991, 279, 1510, 9499, 520, 264, 3813, 10560, 286, 330, 14105, 794, 341, 310, 330, 1337, 794, 330, 1735, 761, 310, 330, 13495, 794, 341, 394, 330, 2588, 794, 341, 504, 330, 1337, 794, 330, 928, 761, 504, 330, 4789, 794, 330, 791, 3813, 311, 636, 279, 9499, 369, 11, 304, 279, 3645, 7393, 13020, 11, 14438, 2153, 702, 394, 457, 310, 1173, 310, 330, 6413, 794, 2330, 394, 330, 2588, 702, 310, 5243, 286, 1173, 286, 330, 693, 794, 341, 310, 330, 1337, 794, 330, 4174, 761, 310, 330, 4789, 794, 330, 791, 1510, 9499, 520, 279, 5300, 3813, 304, 279, 5300, 8316, 11, 439, 264, 2273, 10246, 286, 457, 262, 457, 633, 19182, 11, 1148, 596, 279, 9499, 304, 12366, 1314, 1457, 30, 128009]);
    }
  });
});
