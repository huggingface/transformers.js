export default {
  name: "simple regex",
  responseFormat: {
    type: "regex",
    regex: "[A-Z]{3}-\\d{4}",
  },
  output: "ABC-2026",
};
