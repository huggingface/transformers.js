import { ParserStrategyBase } from './ParserStrategyBase';
import { ParserStrategyGemma4 } from './ParserStrategyGemma4';
import { ParserStrategyGranite } from './ParserStrategyGranite';
import { ParserStrategyQwen3 } from './ParserStrategyQwen3';
import type { ParserContext, ParserStrategy } from './types.ts';

export class ParserRegistry {
    private readonly strategies: ParserStrategy[];

    constructor(strategies?: ParserStrategy[]) {
        this.strategies = strategies ?? [
            new ParserStrategyGemma4(),
            new ParserStrategyGranite(),
            new ParserStrategyQwen3(),
            new ParserStrategyBase(),
        ];
    }

    resolve(context: ParserContext, explicit?: ParserStrategy): ParserStrategy {
        if (explicit) {
            return explicit;
        }

        for (const strategy of this.strategies) {
            if (strategy.supports(context)) {
                return strategy;
            }
        }

        return new ParserStrategyBase();
    }
}
