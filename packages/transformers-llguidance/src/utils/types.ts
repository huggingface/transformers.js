export type GuidanceMask = Uint32Array | Uint8Array | boolean[] | number[];

export type GuidanceMaskResult =
    | { mask: GuidanceMask; vocabSize?: number; stop?: false }
    | { stop: true }
    | { backtrack?: number; ffTokens?: number[] };

export type GuidanceCommitResult = {
    stop?: boolean;
    backtrack?: number;
    ffTokens?: number[];
};

export type GuidanceInterpreter = {
    computeMask(): GuidanceMaskResult;
    commitToken(tokenId: number): GuidanceCommitResult | undefined;
};

export type LlguidanceState = {
    completed: boolean;
    interpreter: GuidanceInterpreter;
    step: number;
};
