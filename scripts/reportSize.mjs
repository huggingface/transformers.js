import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import path from "node:path";
import { colors } from "./logger.mjs";

export const formatSize = (bytes) => {
  if (bytes < 1024) return `${bytes}b`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}kb`;
  return `${(bytes / (1024 * 1024)).toFixed(2)}mb`;
};

export const reportSize = (outfile, log) => {
  const content = readFileSync(outfile);
  const size = content.length;
  const gzipSize = gzipSync(content).length;
  const filename = path.basename(outfile);

  log.done(
    `${colors.bright}${filename}${colors.reset} ${colors.gray}${formatSize(size)}${colors.reset} ${colors.dim}(gzip: ${formatSize(gzipSize)})${colors.reset}`,
  );
};
