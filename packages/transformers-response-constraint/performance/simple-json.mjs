export default {
  name: "simple JSON schema",
  responseFormat: {
    type: "json_schema",
    json_schema: {
      type: "object",
      properties: {
        answer: { type: "string" },
        confidence: { type: "number", minimum: 0, maximum: 1 },
      },
      required: ["answer", "confidence"],
      additionalProperties: false,
    },
  },
  output: '{"answer":"Paris","confidence":0.98}',
};
