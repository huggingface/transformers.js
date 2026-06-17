/**
 * Type definitions for the Cross-Origin Storage API
 * Source: https://github.com/WICG/cross-origin-storage/blob/main/cross-origin-storage.d.ts
 * @see https://github.com/WICG/cross-origin-storage
 */

/**
 * Represents the dictionary for hash algorithm and value.
 */
interface CrossOriginStorageRequestFileHandleHash {
    value: string;
    algorithm: string;
}

/**
 * Represents the options for requesting a file handle.
 */
interface CrossOriginStorageRequestFileHandleOptions {
    create?: boolean;
}

/**
 * The CrossOriginStorageManager interface.
 * [SecureContext]
 */
interface CrossOriginStorageManager {
    requestFileHandle(
        hash: CrossOriginStorageRequestFileHandleHash,
        options?: CrossOriginStorageRequestFileHandleOptions,
    ): Promise<FileSystemFileHandle>;
}

/**
 * Augment the standard Navigator interface.
 */
interface Navigator {
    readonly crossOriginStorage: CrossOriginStorageManager;
}
