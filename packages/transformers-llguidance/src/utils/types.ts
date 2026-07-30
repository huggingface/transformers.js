export type GuidanceMask = Uint32Array | Uint8Array | boolean[] | number[];

export type GuidanceMaskResult =
    | { mask: GuidanceMask; vocabSize?: number; stop?: false }
    | { stop: true }
    | { backtrack?: number; ffTokens?: number[] | Uint32Array };

export type GuidanceCommitResult = {
    stop?: boolean;
    backtrack?: number;
    ffTokens?: number[] | Uint32Array;
};

export type GuidanceInterpreter = {
    computeMask(): GuidanceMaskResult;
    commitToken(tokenId: number): GuidanceCommitResult | undefined;
};

export type LlguidanceStats = {
    steps: number;
    computeMaskMs: number;
    applyMaskMs: number;
    commitTokenMs: number;
};

export type LlguidanceState = {
    completed: boolean;
    interpreter: GuidanceInterpreter;
    step: number;
    stats: LlguidanceStats;
};
