# Improved Download Progress Tracking

## Problem

Transformers.js couldn't reliably track total download progress because:
- File lists weren't known before downloads started
- File sizes were inconsistent (compressed vs uncompressed)
- No cache awareness before initiating downloads

## Solution

### New `ModelRegistry` API
All cache and file management functions are now accessible through a single static class:

```javascript
import { ModelRegistry } from '@huggingface/transformers';

// Check if model is cached
const cached = await ModelRegistry.is_cached('Xenova/bert-base');

// Get all required files
const files = await ModelRegistry.get_files('Xenova/bert-base', { device: 'cpu' });

// Get specific component files
const modelFiles = await ModelRegistry.get_model_files('Xenova/bert-base');
const tokenizerFiles = await ModelRegistry.get_tokenizer_files('Xenova/bert-base');
const processorFiles = await ModelRegistry.get_processor_files('Xenova/bert-base');

// Get file metadata (size, cache status, etc.)
const metadata = await ModelRegistry.get_file_metadata('Xenova/bert-base', 'config.json');
// Returns: { exists: true, size: 794, contentType: 'application/json', fromCache: true }
```

**Methods:**
- **`get_files()`**: Determines all required files (model + tokenizer + processor)
- **`get_model_files()`** / **`get_tokenizer_files()`** / **`get_processor_files()`**: Get files for specific components
- **`get_file_metadata()`**: Fetches metadata using Range requests without downloading
- **`is_cached()`**: Checks if all model files exist in cache

**Key features:**
- Uses HTTP Range requests (`bytes=0-0`) to fetch metadata efficiently
- Returns `fromCache` boolean to distinguish cached vs. remote files
- Ensures consistent **uncompressed** file sizes for accurate progress tracking

### Enhanced Progress Tracking
- **`readResponse()` with `expectedSize`**: Falls back to metadata when `content-length` header is missing
- **`total_progress` callback**: Provides aggregate progress across all files
