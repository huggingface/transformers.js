import { Template } from "@huggingface/jinja";

describe("Jinja templating (chat templates)", () => {
  it("should strip generation/endgeneration tags, including their whitespace-trim variants", () => {
    const template = new Template("Before{%- generation -%}Middle{%- endgeneration -%}After");

    expect(template.render({})).toEqual("BeforeMiddleAfter");
  });
});
