import { ModelAdapterBase } from './ModelAdapterBase';
import { ModelAdapterGemma4 } from './ModelAdapterGemma4';
import { ModelAdapterGranite } from './ModelAdapterGranite';
import { ModelAdapterQwen3 } from './ModelAdapterQwen3';
import type { ModelAdapter, ModelAdapterContext } from './types';

export class ModelAdapterRegistry {
    private readonly adapters: ModelAdapter[];

    constructor(adapters?: ModelAdapter[]) {
        this.adapters = adapters ?? [
            new ModelAdapterGemma4(),
            new ModelAdapterGranite(),
            new ModelAdapterQwen3(),
            new ModelAdapterBase(),
        ];
    }

    resolve(context: ModelAdapterContext): ModelAdapter {
        for (const adapter of this.adapters) {
            if (adapter.supports(context)) {
                return adapter;
            }
        }

        return new ModelAdapterBase();
    }
}
