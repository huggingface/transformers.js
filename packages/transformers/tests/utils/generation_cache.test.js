import { DynamicCache, full } from "../../src/transformers.js";
import { getPastKeyValues } from "../../src/models/modeling_utils.js";
import { jest } from "@jest/globals";

describe("getPastKeyValues", () => {
  const encoderKey = "past_key_values.0.encoder.key";
  const presentEncoderKey = "present.0.encoder.key";

  it("disposes duplicate constant encoder cache outputs", () => {
    const cached = full([1], 1);
    const duplicate = full([1], 2);
    const cache = new DynamicCache({ [encoderKey]: cached });
    const dispose = jest.spyOn(duplicate, "dispose");

    const result = getPastKeyValues({ [presentEncoderKey]: duplicate }, cache);

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(result[encoderKey]).toBe(cached);
    cached.dispose();
  });

  it("does not dispose a reused encoder cache tensor", () => {
    const cached = full([1], 1);
    const cache = new DynamicCache({ [encoderKey]: cached });
    const dispose = jest.spyOn(cached, "dispose");

    const result = getPastKeyValues({ [presentEncoderKey]: cached }, cache);

    expect(dispose).not.toHaveBeenCalled();
    expect(result[encoderKey]).toBe(cached);
    cached.dispose();
  });
});
