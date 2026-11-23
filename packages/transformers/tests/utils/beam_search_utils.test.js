import { selectBeamCandidates } from "../../src/generation/beam_search_utils.js";

const reference = (scores, beam_score, beam_idx, count, tokens = new Map(), penalty = 0) =>
  Array.from(scores, (score, token) => ({ score: beam_score + score - (tokens.get(BigInt(token)) ?? 0) * penalty, token: BigInt(token), beam_idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, count);

describe("Beam candidate selection", () => {
  it.each([1, 2, 8, 50, 1000, 1024, 2000])("matches a full vocabulary sort for top %i", (count) => {
    const scores = Float32Array.from({ length: 1024 }, (_, token) => Math.sin(token * 17) * 10);
    expect(selectBeamCandidates(scores, -3.25, 2, count)).toEqual(reference(scores, -3.25, 2, count));
  });

  it("preserves token order for tied and suppressed scores", () => {
    const scores = new Float32Array([1, -Infinity, 2, 2, -Infinity, 1, 2]);
    for (const count of [2, 3, 5, 7]) expect(selectBeamCandidates(scores, 0, 0, count)).toEqual(reference(scores, 0, 0, count));
  });

  it("applies diversity penalties before selecting candidates", () => {
    const scores = new Float32Array([0, 2, 4, 6, 8]);
    const tokens = new Map([
      [4n, 2],
      [3n, 1],
    ]);
    expect(selectBeamCandidates(scores, -1, 3, 3, tokens, 3)).toEqual(reference(scores, -1, 3, 3, tokens, 3));
  });

  it("handles empty selections", () => {
    expect(selectBeamCandidates(new Float32Array([1, 2]), 0, 0, 0)).toEqual([]);
    expect(selectBeamCandidates(new Float32Array(), 0, 0, 4)).toEqual([]);
  });
});
