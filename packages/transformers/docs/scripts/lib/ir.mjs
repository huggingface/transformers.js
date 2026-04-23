// Group per-file entities (from structure.mjs) into per-module IR. A module is
// one `@module <name>` declaration — its output is `docs/api/<name>.md`.

export function buildIR(fileEntities) {
  const modules = new Map();

  for (const { entities } of fileEntities) {
    const mod = resolveModule(entities);
    if (!mod) continue;
    const entry = modules.get(mod.name) ?? newModule(mod.name);
    if (!entry.description && mod.description) entry.description = mod.description;
    entry.examples.push(...mod.examples);
    ingest(entities, entry);
    modules.set(mod.name, entry);
  }

  return { modules: [...modules.values()], typedefIndex: buildTypedefIndex(modules) };
}

// Classes win (authoritative for a name). Typedefs fill gaps. Re-export
// aliases like `@typedef {import('x').Foo} Foo` register last so they can't
// shadow the canonical definition.
function buildTypedefIndex(modules) {
  const index = new Map();
  for (const mod of modules.values()) {
    for (const cls of mod.classes) index.set(cls.name, mod.name);
  }
  for (const mod of modules.values()) {
    for (const cb of mod.callbacks) index.has(cb.name) || index.set(cb.name, mod.name);
    for (const td of mod.typedefs) {
      if (!isAliasTypedef(td)) index.has(td.name) || index.set(td.name, mod.name);
    }
  }
  for (const mod of modules.values()) {
    for (const td of mod.typedefs) {
      if (isAliasTypedef(td)) index.has(td.name) || index.set(td.name, mod.name);
    }
  }
  return index;
}

function isAliasTypedef(td) {
  if (!td.type) return false;
  return td.type.replace(/import\(['"][^'"]+['"]\)\./g, "").trim() === td.name;
}

function resolveModule(entities) {
  const moduleTag = entities.module?.tags.find((t) => t.tag === "module");
  if (!moduleTag) return null;
  const fileTag = entities.module.tags.find((t) => t.tag === "file");
  return {
    name: moduleTag.name,
    description: fileTag?.description || entities.module.description || "",
    examples: entities.module.tags.filter((t) => t.tag === "example").map(normalizeExample),
  };
}

function newModule(name) {
  return {
    name,
    description: "",
    examples: [],
    classes: [],
    functions: [],
    typedefs: [],
    callbacks: [],
    constants: [],
  };
}

function ingest(entities, mod) {
  for (const block of entities.typedefs) {
    if (!isPrivate(block)) collectTypedefsFromBlock(block, mod);
  }
  for (const cls of entities.classes) {
    if (isPrivate(cls)) continue;
    mod.classes.push({
      name: cls.name,
      description: cls.description,
      examples: tagsOf(cls, "example").map(normalizeExample),
      skillExamples: tagsOf(cls, "skillExample").map((t) => t.task),
      members: cls.members.filter((m) => !isPrivate(m)).map(buildMember),
    });
  }
  for (const fn of entities.functions) {
    if (!isPrivate(fn)) mod.functions.push(buildCallable(fn));
  }
  for (const v of entities.variables) {
    if (isPrivate(v)) continue;
    mod.constants.push({
      name: v.name,
      type: typeOf(v),
      description: v.description,
      examples: tagsOf(v, "example").map(normalizeExample),
    });
  }
}

// A single @typedef/@callback block may declare multiple types; @property tags
// that follow each typedef attach to it, up to the next typedef/callback.
function collectTypedefsFromBlock(block, mod) {
  let current = null;
  let first = true;
  const flush = () => {
    if (!current) return;
    if (current.kind === "typedef") mod.typedefs.push(current);
    else mod.callbacks.push(current);
    current = null;
  };
  for (const tag of block.tags) {
    if (tag.tag === "typedef") {
      flush();
      current = {
        kind: "typedef",
        name: tag.name,
        type: tag.type,
        description: tag.description || (first ? block.description : ""),
        properties: [],
      };
      first = false;
    } else if (tag.tag === "callback") {
      flush();
      current = {
        kind: "callback",
        name: tag.name,
        description: block.description,
        params: tagsOf(block, "param").map(normalizeParam),
        returns: pickReturns(block),
      };
      first = false;
    } else if ((tag.tag === "property" || tag.tag === "prop") && current?.kind === "typedef") {
      current.properties.push(normalizeParam(tag));
    }
  }
  flush();
}

function buildMember(m) {
  if (m.kind === "method") return { ...buildCallable(m), kind: "method" };
  return {
    kind: m.kind,
    name: m.name,
    description: m.description,
    type: typeOf(m),
    defaultValue: m.tags.find((t) => t.tag === "default")?.value ?? m.initializer ?? null,
    deprecated: m.tags.some((t) => t.tag === "deprecated"),
  };
}

function buildCallable(fn) {
  return {
    name: fn.name,
    description: fn.description,
    params: tagsOf(fn, "param").map(normalizeParam),
    returns: pickReturns(fn),
    throws: tagsOf(fn, "throws").map((t) => ({ type: t.type, description: t.description })),
    examples: tagsOf(fn, "example").map(normalizeExample),
    skillExamples: tagsOf(fn, "skillExample").map((t) => t.task),
    deprecated: fn.tags.some((t) => t.tag === "deprecated"),
  };
}

function tagsOf(entity, name) {
  return entity.tags.filter((t) => t.tag === name);
}

function typeOf(entity) {
  return entity.tags.find((t) => t.tag === "type")?.type ?? null;
}

function pickReturns(entity) {
  const r = entity.tags.find((t) => t.tag === "returns");
  return r ? { type: r.type, description: r.description || "" } : null;
}

function normalizeParam(tag) {
  return {
    name: tag.name,
    type: tag.type,
    optional: !!tag.optional,
    defaultValue: tag.defaultValue ?? null,
    description: tag.description || "",
  };
}

const FENCE_BLOCK = /```(\w+)?\n([\s\S]*?)\n```/;
const EXAMPLE_PREFIX = /^(?:\*\*Example[^*]*\*\*|Example)[:\s]*/;

function normalizeExample(tag) {
  const body = tag.body || "";
  const m = body.match(FENCE_BLOCK);
  if (!m) return { title: "", language: "javascript", code: body.trim() };
  return {
    title: body.slice(0, m.index).trim().replace(EXAMPLE_PREFIX, "").trim(),
    language: m[1] || "javascript",
    code: m[2].trim(),
  };
}

function isPrivate(entity) {
  return entity.tags?.some((t) => t.tag === "private");
}
