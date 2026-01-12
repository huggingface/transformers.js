# Model Registry Architecture

## Overview

This document explains how the model registry system works in transformers.js and how we solved the circular dependency problem with **automatic derivation** of validation mappings.

## The Problem

Previously, we had a circular dependency and duplicate maintenance:

```
modeling_utils.js 
  ↓ imports
model-mapping-names.js (name-only mappings)
  ↑ imports from
registry.js (full mappings with classes)
  ↑ imports from  
modeling_utils.js (MODEL_TYPES, Maps to populate)
```

This created maintenance burden:
- Two separate mapping files to maintain
- **Duplicate model name definitions** (once with classes, once without)
- Risk of inconsistencies between the two files
- Every new model required updates in both places

## The Solution

We use **automatic derivation** - maintain mappings once in registry.js, automatically derive name-only versions for validation:

```
modeling_utils.js
  ├─ Defines MODEL_TYPES enum
  ├─ Defines empty Maps: MODEL_TYPE_MAPPING, MODEL_NAME_TO_CLASS_MAPPING, MODEL_CLASS_TO_NAME_MAPPING
  ├─ Defines null placeholders for validation mappings (e.g., MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
  ├─ Exports helper: toNameOnlyMapping() to convert full mappings to name-only
  ├─ Exports registerTaskMappings() to populate validation mappings
  └─ Exports PreTrainedModel base class

registry.js
  ├─ Imports from modeling_utils.js (one-way dependency ✓)
  ├─ Imports all model classes
  ├─ Defines full mappings with class references (SINGLE SOURCE OF TRUTH)
  ├─ Populates the Maps from modeling_utils.js
  └─ Calls registerTaskMappings() to auto-derive name-only versions
```

### Key Design Principles

1. **Single Source of Truth**: Model mappings are defined **once and only once** in `registry.js`

2. **Automatic Derivation**: Name-only mappings for validation are automatically derived from full mappings
   - No manual duplication
   - No risk of inconsistency
   - One place to update when adding models

3. **Clear Dependency Direction**: 
   - `registry.js` imports from `modeling_utils.js` ✓
   - `modeling_utils.js` does NOT import from `registry.js` ✓
   - No circular dependency ✓

4. **Separation of Concerns**:
   - **modeling_utils.js**: Base functionality, validation logic (needs names only)
   - **registry.js**: Registration and class references (needs actual classes)

## File Responsibilities

### `modeling_utils.js`

**Purpose**: Core model utilities and base classes

**Exports**:
- `MODEL_TYPES` - Enum of model architecture types
- `MODEL_TYPE_MAPPING` - Map<string, number> (empty, populated by registry.js)
- `MODEL_NAME_TO_CLASS_MAPPING` - Map<string, Function> (empty, populated by registry.js)
- `MODEL_CLASS_TO_NAME_MAPPING` - Map<Function, string> (empty, populated by registry.js)
- Name-only mappings for validation (null initially, auto-derived by registry.js):
  - `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` - Map<modelType, [className]>
  - `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES`
  - `MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES`
  - `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`
- `toNameOnlyMapping(mapping)` - Helper to convert full mapping to name-only
- `registerTaskMappings(mappings)` - Called by registry.js to populate validation mappings
- `PreTrainedModel` - Base class for all models

**Usage**:
- Model validation (`_validate_model_class()` only needs class names, not references)
- Determining model architecture type
- Base class for model implementations

### `registry.js`

**Purpose**: Register all model classes and populate mappings

**Exports**:
- All model classes (e.g., `BertModel`, `GPT2Model`, etc.)
- Full mappings with class references:
  - `MODEL_MAPPING_NAMES_ENCODER_ONLY` - Map<modelType, [className, ClassRef]>
  - `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` - Map<modelType, [className, ClassRef]>
  - etc.

**Responsibility**:
- Import all model classes from `./pre-trained-models/index.js`
- Define mappings that include class references
- Populate the Maps exported from `modeling_utils.js`

## How It Works

### 1. Initialization (modeling_utils.js)

```javascript
// modeling_utils.js defines empty structures
export const MODEL_TYPE_MAPPING = new Map();  // Empty, to be populated
export let MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = null;  // Null, to be auto-derived

// Helper to convert full mapping to name-only
export function toNameOnlyMapping(mapping) {
  const nameOnly = new Map();
  for (const [key, value] of mapping) {
    nameOnly.set(key, [value[0]]);  // Keep only name, drop class reference
  }
  return nameOnly;
}

// Called by registry.js to populate validation mappings
export function registerTaskMappings(mappings) {
  MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = toNameOnlyMapping(mappings.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES);
  // ... same for other mappings
}
```

### 2. Registration (registry.js)

```javascript
// registry.js imports and defines FULL mappings (single source of truth)
import {
  MODEL_TYPE_MAPPING,
  MODEL_TYPES,
  registerTaskMappings
} from './modeling_utils.js';
import { LlamaForCausalLM } from './pre-trained-models/index.js';

// Define full mappings with class references (SINGLE SOURCE OF TRUTH)
export const MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = new Map([
  ['llama', ['LlamaForCausalLM', LlamaForCausalLM]],  // Name AND class reference
  // ...
]);

// Populate the global maps
for (const [mappings, type] of MODEL_CLASS_TYPE_MAPPING) {
  for (const [name, model] of mappings.values()) {
    MODEL_TYPE_MAPPING.set(name, type);
    MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
  }
}

// Auto-derive and register name-only versions for validation
registerTaskMappings({
  MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
  MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
  // ...
});
```

### 3. Usage

```javascript
// In PreTrainedModel constructor (modeling_utils.js)
constructor(config, sessions, configs) {
  const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);
  const modelType = MODEL_TYPE_MAPPING.get(modelName);
  // ... use modelType to configure behavior
}

// In validation method (modeling_utils.js)
_validate_model_class() {
  const modelType = this.config.model_type;
  const supported_models = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(modelType);
  if (supported_models) {
    // supported_models[0] is the class name string
    generate_compatible_classes.add(supported_models[0]);
  }
}
```

## Benefits

1. ✅ **No Circular Dependencies**: Clear one-way import flow
2. ✅ **No Duplicate Maintenance**: Model names defined **once and only once** in registry.js
3. ✅ **Automatic Consistency**: Name-only mappings are derived automatically, impossible to get out of sync
4. ✅ **Single Point of Update**: Add a model in one place (registry.js), validation mappings update automatically
5. ✅ **Type Safety**: Maps are properly typed and populated
6. ✅ **Extensibility**: Easy to add new models - just add to registry.js
7. ✅ **Clear Separation**: Validation logic doesn't need class references

## Adding a New Model

To add a new model to the registry, you only need to update **ONE place**:

1. **Create the model class** in `src/models/pre-trained-models/your-model.js`

2. **Export from index** in `src/models/pre-trained-models/index.js`:
   ```javascript
   export { YourModel, YourForCausalLM } from './your-model.js';
   ```

3. **Add to appropriate mapping in registry.js** (ONLY HERE - the single source of truth):
   ```javascript
   import { YourModel, YourForCausalLM } from './pre-trained-models/index.js';
   
   const MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = new Map([
     // ...
     ['your-model', ['YourForCausalLM', YourForCausalLM]],  // Name AND class
   ]);
   ```

That's it! 

✨ **The name-only version for validation is automatically derived** - no need to update modeling_utils.js!

The registration logic in registry.js will:
- Populate the global Maps (MODEL_TYPE_MAPPING, etc.)
- Automatically derive name-only versions by calling `registerTaskMappings()`
- Make validation mappings available in modeling_utils.js

## Troubleshooting

### Circular Dependency Error

If you see a circular dependency error:
- Check that `modeling_utils.js` does NOT import from `registry.js`
- Ensure `modeling_utils.js` only exports empty Maps and name-only mappings
- Verify `registry.js` is the one importing and populating

### Model Not Found

If a model isn't being recognized:
1. Check it's exported from `pre-trained-models/index.js`
2. Verify it's added to the appropriate mapping in `registry.js`
3. Ensure the model type string matches between config and mapping key

### Type Mismatch

If model type detection fails:
1. Verify `MODEL_CLASS_TYPE_MAPPING` in registry.js includes your mapping
2. Check that the MODEL_TYPES value is correct
3. Ensure the model class is properly registered in the mapping
