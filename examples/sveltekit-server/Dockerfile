FROM node:18 AS base

RUN apt-get update && \
    apt-get install -y tini

ENTRYPOINT ["/usr/bin/tini", "--"]

# Install dependencies only when needed
FROM base AS deps

WORKDIR /app

# Install dependencies based on the preferred package manager
COPY --link package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps --link /app/node_modules ./node_modules
COPY --link  . .

RUN npm run build && npm prune --production

ENV NODE_ENV=production

COPY --link package.json .
EXPOSE 3000


# Allow the running process to write model files to the cache folder.
# NOTE: In practice, you would probably want to pre-download the model files to avoid having to download them on-the-fly.
RUN mkdir -p /app/node_modules/@xenova/.cache/
RUN chmod 777 -R /app/node_modules/@xenova/

CMD ["node", "build"]