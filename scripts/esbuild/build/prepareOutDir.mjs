import { existsSync, mkdirSync, rmSync } from "node:fs";

export default function prepareOutDir(dir) {
  if (existsSync(dir)) {
    rmSync(dir, { recursive: true, force: true });
  }

  mkdirSync(dir, { recursive: true });
}
