# Overall Progress Tracking Example

The `pipeline()` function now provides overall progress information in addition to per-file progress.

## Enhanced Progress Information

When using `progress_callback` with `pipeline()`, you now receive:

- **Per-file progress**: `progress`, `loaded`, `total` (as before)
- **Overall progress**: `totalLoaded`, `totalBytes`, `totalProgress` (NEW!)

## Progress Info Fields

Progress events now include:

```typescript
{
  status: 'initiate' | 'download' | 'progress' | 'done' | 'ready',
  name: string,           // model id
  file?: string,          // current file (not present in 'ready')
  
  // Per-file progress (for 'progress' status)
  progress?: number,      // 0-100 for current file
  loaded?: number,        // bytes loaded for current file
  total?: number,         // total bytes for current file
  
  // Overall progress (NEW!)
  totalLoaded?: number,   // total bytes loaded across all files
  totalBytes?: number,    // total bytes to load across all files
  totalProgress?: number, // 0-100 for overall progress
}
```

## Usage Example

### Basic Progress Bar

```javascript
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', {
    progress_callback: (info) => {
        if (info.status === 'progress') {
            // Per-file progress
            console.log(`${info.file}: ${info.progress.toFixed(1)}% (${formatBytes(info.loaded)}/${formatBytes(info.total)})`);
            
            // Overall progress
            console.log(`Overall: ${info.totalProgress.toFixed(1)}% (${formatBytes(info.totalLoaded)}/${formatBytes(info.totalBytes)})`);
        } else if (info.status === 'ready') {
            console.log(`✓ Model loaded! Total size: ${formatBytes(info.totalLoaded)}`);
        }
    }
});

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
```

### React Progress Component

```jsx
import { useState, useEffect } from 'react';
import { pipeline } from '@huggingface/transformers';

function ModelLoader() {
    const [progress, setProgress] = useState({
        status: 'idle',
        currentFile: '',
        fileProgress: 0,
        totalProgress: 0,
        totalLoaded: 0,
        totalBytes: 0,
    });
    
    useEffect(() => {
        async function loadModel() {
            const classifier = await pipeline(
                'sentiment-analysis',
                'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                {
                    progress_callback: (info) => {
                        setProgress({
                            status: info.status,
                            currentFile: info.file || '',
                            fileProgress: info.progress || 0,
                            totalProgress: info.totalProgress || 0,
                            totalLoaded: info.totalLoaded || 0,
                            totalBytes: info.totalBytes || 0,
                        });
                    }
                }
            );
            // Use classifier...
        }
        
        loadModel();
    }, []);
    
    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    return (
        <div>
            <h3>Loading Model...</h3>
            
            {/* Overall progress bar */}
            <div>
                <div>
                    Overall: {formatBytes(progress.totalLoaded)} / {formatBytes(progress.totalBytes)}
                </div>
                <progress value={progress.totalProgress} max={100} />
                <span>{progress.totalProgress.toFixed(0)}%</span>
            </div>
            
            {/* Current file progress */}
            {progress.currentFile && (
                <div>
                    <div>Loading: {progress.currentFile}</div>
                    <progress value={progress.fileProgress} max={100} />
                    <span>{progress.fileProgress.toFixed(0)}%</span>
                </div>
            )}
            
            {progress.status === 'ready' && (
                <div>✓ Model ready!</div>
            )}
        </div>
    );
}
```

### Terminal Progress Indicator

```javascript
import { pipeline } from '@huggingface/transformers';

let lastOverallPercent = 0;

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', {
    progress_callback: (info) => {
        if (info.status === 'initiate') {
            console.log(`📥 Downloading ${info.file}...`);
        }
        
        if (info.status === 'done') {
            console.log(`✓ ${info.file} complete`);
        }
        
        if (info.status === 'progress' && info.totalProgress !== undefined) {
            const overallPercent = Math.floor(info.totalProgress);
            
            // Only log when percentage changes
            if (overallPercent > lastOverallPercent) {
                lastOverallPercent = overallPercent;
                const bar = '█'.repeat(Math.floor(overallPercent / 2));
                const empty = '░'.repeat(50 - Math.floor(overallPercent / 2));
                const loaded = formatBytes(info.totalLoaded);
                const total = formatBytes(info.totalBytes);
                console.log(`[${bar}${empty}] ${overallPercent}% (${loaded}/${total})`);
            }
        }
        
        if (info.status === 'ready') {
            console.log(`🎉 Model ready! Loaded ${formatBytes(info.totalLoaded)}.`);
        }
    }
});
```

## Status Flow

1. **`initiate`** - File download starts
   - Includes: `name`, `file`

2. **`download`** - File is being downloaded
   - Includes: `name`, `file`

3. **`progress`** - Download progress update
   - Per-file: `name`, `file`, `progress`, `loaded`, `total`
   - Overall: `totalLoaded`, `totalBytes`, `totalProgress`

4. **`done`** - File download complete
   - Includes: `name`, `file`, `totalLoaded`, `totalBytes`, `totalProgress`

5. **`ready`** - All files loaded, model ready
   - Includes: `task`, `model`, `totalLoaded`

## Using Overall Progress

The overall progress is **automatically calculated** and included in the progress info:

```javascript
progress_callback: (info) => {
    if (info.status === 'progress') {
        // Already calculated for you!
        console.log(`Overall: ${info.totalProgress.toFixed(1)}%`);
        console.log(`Loaded: ${info.totalLoaded} / ${info.totalBytes} bytes`);
    }
}
```

### How It's Calculated

The pipeline automatically:
1. Tracks each file's total size when the first progress event arrives
2. Sums completed files' sizes
3. Adds the current file's loaded bytes
4. Calculates: `(totalLoaded / totalBytes) * 100`

This gives you **smooth, byte-accurate progress** that accounts for different file sizes.

## Benefits

- ✅ **Accurate progress**: Weighted by actual file sizes, not just file count
- ✅ **Smooth progress bars**: No sudden jumps between files
- ✅ **Better UX**: Show "45.2MB / 120MB" instead of just "Loading..."
- ✅ **ETA calculations**: Calculate time remaining based on bytes/sec
- ✅ **Automatic**: No manual tracking needed - it's all calculated for you
- ✅ **Debugging**: See if progress stalls on a particular file

## Notes

- **Byte-accurate**: Progress is weighted by actual file sizes
- **Smooth updates**: Progress updates continuously, not in discrete steps
- **Automatic calculation**: You don't need to track anything manually
- **Cached files**: Files loaded from cache also trigger progress events
- **File sizes known on first progress**: `totalBytes` becomes available after first file starts downloading
- **Ready event**: Final event includes total bytes loaded

## Example Output

```
📥 Downloading tokenizer.json...
[██░░░░░░░░░░░░░░░░░░░░░░░░] 4% (1.2MB/28.5MB)
[█████░░░░░░░░░░░░░░░░░░░░░] 10% (2.9MB/28.5MB)
✓ tokenizer.json complete
📥 Downloading onnx/model.onnx...
[███████░░░░░░░░░░░░░░░░░░░] 14% (4.1MB/28.5MB)
[████████████░░░░░░░░░░░░░░] 24% (6.8MB/28.5MB)
[████████████████████░░░░░░] 40% (11.4MB/28.5MB)
[████████████████████████████████] 64% (18.2MB/28.5MB)
✓ onnx/model.onnx complete
[██████████████████████████████████████████████████] 100% (28.5MB/28.5MB)
🎉 Model ready! Loaded 28.5 MB.
```
