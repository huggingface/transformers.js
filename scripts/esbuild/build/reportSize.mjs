import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

export const formatSize = (bytes) => {
  if (bytes < 1024) return `${bytes}b`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}kb`;
  return `${(bytes / (1024 * 1024)).toFixed(2)}mb`;
};

export const reportSize = (outfile) => {
  const content = readFileSync(outfile);
  const size = content.length;
  const gzipSize = gzipSync(content).length;

  console.log(`\n${outfile}\n${formatSize(size)} (gzip: ${formatSize(gzipSize)})`);
};
