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

export const log = {
  section: (text) => console.log(`\n${colors.bright}${colors.cyan}=== ${text} ===${colors.reset}`),
  info: (text) => console.log(`${colors.blue}[info]${colors.reset} ${text}`),
  success: (text) => console.log(`${colors.green}✓${colors.reset} ${text}`),
  warning: (text) => console.log(`${colors.yellow}[warn]${colors.reset} ${text}`),
  error: (text) => console.log(`${colors.red}[error]${colors.reset} ${text}`),
  dim: (text) => console.log(`${colors.dim}${text}${colors.reset}`),
  url: (text) => console.log(`  ${colors.cyan}→${colors.reset} ${colors.bright}${text}${colors.reset}`),
  file: (text) => console.log(`  ${colors.gray}-${colors.reset} ${text}`),
  build: (text) => console.log(`${colors.cyan}[build]${colors.reset} ${text}`),
  done: (text) => console.log(`${colors.green}[done]${colors.reset} ${text}`),
};
