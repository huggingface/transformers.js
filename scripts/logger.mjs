/**
 * Shared logging utility with colored output
 */

// ANSI color codes
export const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  red: "\x1b[31m",
  gray: "\x1b[90m",
};

/**
 * Creates a logger with optional package prefix
 * @param {string} [prefix] - Optional prefix to prepend to all logs (e.g., "transformers", "react")
 * @returns {Object} Logger object with various log methods
 */
export function createLogger(prefix = "") {
  const formatPrefix = prefix ? `${colors.magenta}[${prefix}]${colors.reset} ` : "";

  return {
    section: (text) => console.log(`\n${formatPrefix}${colors.bright}${colors.cyan}=== ${text} ===${colors.reset}`),
    info: (text) => console.log(`${formatPrefix}${colors.blue}[info]${colors.reset} ${text}`),
    success: (text) => console.log(`${formatPrefix}${colors.green}✓${colors.reset} ${text}`),
    warning: (text) => console.log(`${formatPrefix}${colors.yellow}[warn]${colors.reset} ${text}`),
    error: (text) => console.log(`${formatPrefix}${colors.red}[error]${colors.reset} ${text}`),
    dim: (text) => console.log(`${formatPrefix}${colors.dim}${text}${colors.reset}`),
    url: (text) => console.log(`${formatPrefix}  ${colors.cyan}→${colors.reset} ${colors.bright}${text}${colors.reset}`),
    file: (text) => console.log(`${formatPrefix}  ${colors.gray}-${colors.reset} ${text}`),
    build: (text) => console.log(`${formatPrefix}${colors.cyan}[build]${colors.reset} ${text}`),
    done: (text) => console.log(`${formatPrefix}${colors.green}[done]${colors.reset} ${text}`),
  };
}

// Default logger without prefix for backward compatibility
export const log = createLogger();
