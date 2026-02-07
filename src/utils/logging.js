/**
 * @file Logging utilities for Transformers.js, modeled after the Python
 * `transformers.utils.logging` module.
 *
 * **Example:** Set the log level to WARNING (hide info messages).
 * ```javascript
 * import { logging } from '@huggingface/transformers';
 * logging.setLogLevel('WARNING');
 * ```
 *
 * **Example:** Get a logger for a specific module.
 * ```javascript
 * import { logging } from '@huggingface/transformers';
 * const logger = logging.getLogger('my-module');
 * logger.info('This is an info message');
 * ```
 *
 * @module utils/logging
 */

/**
 * @enum {number}
 * @readonly
 */
export const LoggingLevel = Object.freeze({
  DEBUG: 0,
  INFO: 1,
  WARNING: 2,
  ERROR: 3,
  /** Disable all logging. */
  SILENT: 4,
});

/**
 * Mapping from string names to logging level values.
 * @type {Record<string, number>}
 * @private
 */
const LOG_LEVEL_NAMES = Object.freeze({
  debug: LoggingLevel.DEBUG,
  info: LoggingLevel.INFO,
  warning: LoggingLevel.WARNING,
  warn: LoggingLevel.WARNING, // alias
  error: LoggingLevel.ERROR,
  silent: LoggingLevel.SILENT,
});

/** @type {number} */
let _logLevel = LoggingLevel.WARNING;

/**
 * Set the global logging level.
 *
 * @param {number|string} level The logging level. Can be a numeric value from
 *   {@link LoggingLevel} or one of the strings: `'DEBUG'`, `'INFO'`,
 *   `'WARNING'`, `'ERROR'`, `'SILENT'`.
 */
export function setLogLevel(level) {
  if (typeof level === "string") {
    const resolved = LOG_LEVEL_NAMES[level.toLowerCase()];
    if (resolved === undefined) {
      throw new Error(
        `Unknown log level: "${level}". ` +
          `Valid levels are: ${Object.keys(LOG_LEVEL_NAMES).join(", ")}`,
      );
    }
    _logLevel = resolved;
  } else if (typeof level === "number") {
    _logLevel = level;
  } else {
    throw new Error(
      `Invalid log level type: ${typeof level}. Expected string or number.`,
    );
  }
}

/**
 * Get the current global logging level.
 * @returns {number} The current logging level.
 */
export function getLogLevel() {
  return _logLevel;
}

/**
 * Set the global logging level to DEBUG.
 */
export function setLogLevelDebug() {
  setLogLevel(LoggingLevel.DEBUG);
}

/**
 * Set the global logging level to INFO.
 */
export function setLogLevelInfo() {
  setLogLevel(LoggingLevel.INFO);
}

/**
 * Set the global logging level to WARNING (the default).
 */
export function setLogLevelWarning() {
  setLogLevel(LoggingLevel.WARNING);
}

/**
 * Set the global logging level to ERROR.
 */
export function setLogLevelError() {
  setLogLevel(LoggingLevel.ERROR);
}

/**
 * A simple logger that respects the global logging level.
 */
class Logger {
  /**
   * @param {string} [name=''] An optional name for this logger (used as a prefix in messages).
   */
  constructor(name = "") {
    /** @type {string} */
    this.name = name;
  }

  /**
   * @param {string} method The console method to call.
   * @param {any[]} args The arguments to pass.
   * @private
   */
  _log(method, ...args) {
    if (this.name) {
      console[method](`[${this.name}]`, ...args);
    } else {
      console[method](...args);
    }
  }

  /**
   * Log a debug message (only shown when log level <= DEBUG).
   * @param  {...any} args
   */
  debug(...args) {
    if (_logLevel <= LoggingLevel.DEBUG) {
      this._log("debug", ...args);
    }
  }

  /**
   * Log an info message (only shown when log level <= INFO).
   * @param  {...any} args
   */
  info(...args) {
    if (_logLevel <= LoggingLevel.INFO) {
      this._log("info", ...args);
    }
  }

  /**
   * Log a warning message (only shown when log level <= WARNING).
   * @param  {...any} args
   */
  warn(...args) {
    if (_logLevel <= LoggingLevel.WARNING) {
      this._log("warn", ...args);
    }
  }

  /**
   * Log an error message (only shown when log level <= ERROR).
   * @param  {...any} args
   */
  error(...args) {
    if (_logLevel <= LoggingLevel.ERROR) {
      this._log("error", ...args);
    }
  }
}

/** @type {Map<string, Logger>} */
const _loggers = new Map();

/**
 * Get (or create) a logger for the given name.
 *
 * @param {string} [name=''] The name of the logger.
 * @returns {Logger} A Logger instance.
 */
export function getLogger(name = "") {
  let logger = _loggers.get(name);
  if (!logger) {
    logger = new Logger(name);
    _loggers.set(name, logger);
  }
  return logger;
}

/**
 * A convenience namespace that groups all logging utilities, similar to
 * Python's `transformers.utils.logging`.
 *
 * @example
 * import { logging } from '@huggingface/transformers';
 * logging.setLogLevel('INFO');
 * const logger = logging.getLogger('my-module');
 * logger.info('hello');
 */
export const logging = Object.freeze({
  LoggingLevel,
  setLogLevel,
  getLogLevel,
  setLogLevelDebug,
  setLogLevelInfo,
  setLogLevelWarning,
  setLogLevelError,
  getLogger,
});
