# Proposal: portable execution manifest for inference providers

> [!NOTE]
> This is a design proposal, not an implemented contract. Nothing described here is consumed by Transformers.js today. It is kept separate from the [Inference Runtime Integration Guide](INFERENCE_RUNTIME_INTEGRATION_GUIDE.md) so that readers of the guide can tell what exists from what is planned.

## Motivation

The experimental normalized-session contract removes per-model runtime loaders: Transformers.js selects the built-in model class, determines the required session roles, and asks the provider to compile one graph per role. But graph files alone do not describe which graph is an encoder, decoder, embedding model, cache updater, or task head. They also do not describe graph variants, named state bindings, multimodal routing, or generation behavior. Today those semantics come from the built-in ONNX session configuration, which ties the contract to ONNX file-naming conventions.

A portable execution manifest should describe those semantics without exposing nodes from one graph format as the Transformers.js public API.

## Proposed first version

The manifest is repository metadata consumed by Transformers.js and passed to any compatible provider:

```ts
type ExecutionManifestV1 = {
    version: 1;
    modelType?: string;
    artifacts: Record<
        string,
        {
            file: string;
            format: string;
            role: string;
            variants?: Array<{
                file: string;
                dtype?: string;
                device?: string;
            }>;
            externalData?: string[];
        }
    >;
    sessions: Record<
        string,
        {
            artifact: string;
            inputs?: Record<string, string>;
            outputs?: Record<string, string>;
            state?: Array<{
                input: string;
                output: string;
                axis?: number;
            }>;
        }
    >;
    orchestration: {
        kind: 'single' | 'encoder-decoder' | 'causal-decoder' | 'multimodal';
        entrypoints: Record<string, string>;
    };
    extensions?: Record<string, unknown>;
};
```

Artifact entries describe files and selectable variants. Session entries map semantic roles to artifacts and define logical input, output, and persistent-state bindings. Orchestration identifies a versioned Transformers.js execution protocol rather than embedding runtime-specific control flow. Providers remain responsible for compiling the selected artifact and may reject unsupported formats, devices, data types, or orchestration kinds.

## Compatibility rules

- `version` changes only for incompatible structural changes;
- graph format names and provider extensions are open strings;
- semantic roles and orchestration kinds are versioned by Transformers.js;
- unknown optional fields are ignored, while unknown required orchestration kinds are rejected;
- file selection uses the same resolved manifest in loading, progress calculation, cache inspection, and deletion;
- runtime-specific settings live under namespaced `extensions`, not in common session semantics;
- tensor ownership and state lifetime use the public backend-storage contract rather than graph-format handles.

## Implementation order

The first implementation target should be `single`, followed by `causal-decoder` using the existing autoregressive-session protocol. `encoder-decoder` and `multimodal` should be added only after their graph roles, state transitions, and input-binding rules are represented without architecture-specific callbacks.

Until this manifest is implemented, normalized session providers use the existing Transformers.js model configuration and session-role mappings, and the normalized-session contract stays experimental.
