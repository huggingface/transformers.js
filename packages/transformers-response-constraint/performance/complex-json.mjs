export default {
  name: "complex JSON schema",
  responseFormat: {
    type: "json_schema",
    json_schema: {
      type: "object",
      properties: {
        request_id: { type: "string", pattern: "^[a-z0-9-]{8,36}$" },
        status: { enum: ["queued", "running", "completed", "failed"] },
        user: {
          type: "object",
          properties: {
            id: { type: "integer", minimum: 1 },
            email: { type: "string", format: "email" },
            roles: {
              type: "array",
              items: { enum: ["admin", "editor", "viewer"] },
              minItems: 1,
              uniqueItems: true,
            },
          },
          required: ["id", "email", "roles"],
          additionalProperties: false,
        },
        results: {
          type: "array",
          minItems: 2,
          maxItems: 8,
          items: {
            type: "object",
            properties: {
              label: { type: "string", minLength: 2, maxLength: 32 },
              score: { type: "number", minimum: 0, maximum: 1 },
              tags: {
                type: "array",
                items: { type: "string", pattern: "^[a-z-]+$" },
                maxItems: 5,
              },
              metadata: {
                anyOf: [
                  { type: "null" },
                  {
                    type: "object",
                    properties: {
                      source: { type: "string" },
                      cached: { type: "boolean" },
                    },
                    required: ["source", "cached"],
                    additionalProperties: false,
                  },
                ],
              },
            },
            required: ["label", "score", "tags", "metadata"],
            additionalProperties: false,
          },
        },
      },
      required: ["request_id", "status", "user", "results"],
      additionalProperties: false,
    },
  },
  output:
    '{"request_id":"req-2026-a1","status":"completed","user":{"id":42,"email":"user@example.com","roles":["admin","editor"]},"results":[{"label":"primary","score":0.97,"tags":["fast","verified"],"metadata":{"source":"cache-v2","cached":true}},{"label":"fallback","score":0.81,"tags":["review"],"metadata":null}]}',
};
