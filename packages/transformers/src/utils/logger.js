/**
 * @file Logger utility for Transformers.js with configurable log levels.
 *
 * @module utils/logger
 */

import { env, LogLevel } from '../env.js';

/**
 * Logger that respects the configured log level in env.logLevel.
 *
 * @example
 * import { logger } from './utils/logger.js';
 * logger.info('Model loaded successfully');
 * logger.warn('Deprecated method used');
 * logger.error('Failed to load model');
 * logger.debug('Token count:', tokens.length);
 */
export const logger = {
    /**
     * Log an error message. Only suppressed when logLevel is NONE.
     * @param  {...any} args - Arguments to log
     */
    error(...args) {
        if (env.logLevel <= LogLevel.ERROR) {
            console.error(...args);
        }
    },

    /**
     * Log a warning message. Shown when logLevel <= WARNING.
     * @param  {...any} args - Arguments to log
     */
    warn(...args) {
        if (env.logLevel <= LogLevel.WARNING) {
            console.warn(...args);
        }
    },

    /**
     * Log an info message. Shown when logLevel <= INFO.
     * @param  {...any} args - Arguments to log
     */
    info(...args) {
        if (env.logLevel <= LogLevel.INFO) {
            console.log(...args);
        }
    },

    /**
     * Log a debug message. Only shown when logLevel is DEBUG.
     * @param  {...any} args - Arguments to log
     */
    debug(...args) {
        if (env.logLevel <= LogLevel.DEBUG) {
            console.log(...args);
        }
    },

    /**
     * Log a message (alias for info). Shown when logLevel <= INFO.
     * @param  {...any} args - Arguments to log
     */
    log(...args) {
        this.info(...args);
    },
};
