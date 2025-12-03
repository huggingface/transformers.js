import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

export const startServer = (dir, PORT = 8080) =>
  new Promise((resolve) => {
    const server = createServer((req, res) => {
      // Enable CORS
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type");

      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }

      let filePath = req.url === "/" ? "/index.html" : req.url;
      filePath = filePath.split("?")[0]; // Remove query params

      // Try to serve from outdir first, then fall back to rootDir
      let fullPath = path.join(dir, filePath);

      // Check if file exists
      if (!existsSync(fullPath)) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("404 Not Found");
        return;
      }

      // Check if it's a directory
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        fullPath = path.join(fullPath, "index.html");
        if (!existsSync(fullPath)) {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("404 Not Found");
          return;
        }
      }

      // Get MIME type
      const ext = path.extname(fullPath);
      const mimeType = MIME_TYPES[ext] || "application/octet-stream";

      try {
        const content = readFileSync(fullPath);
        res.writeHead(200, { "Content-Type": mimeType });
        res.end(content);
      } catch (error) {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end("500 Internal Server Error");
      }
    });

    server.listen(PORT, () => {
      resolve(server);
    });
  });
