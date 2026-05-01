import { BaseParserStrategy } from './BaseParserStrategy';
import { Gemma4ParserStrategy } from './Gemma4ParserStrategy';
import { GraniteParserStrategy } from './GraniteParserStrategy';
import { Qwen3ParserStrategy } from './Qwen3ParserStrategy';
import type { ParserContext, ParserStrategy } from './types.ts';

export class ParserRegistry {
    private readonly strategies: ParserStrategy[];

    constructor(strategies?: ParserStrategy[]) {
        this.strategies = strategies ?? [
            new Gemma4ParserStrategy(),
            new GraniteParserStrategy(),
            new Qwen3ParserStrategy(),
            new BaseParserStrategy(),
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

        return new BaseParserStrategy();
    }
}
