// Group per-file entities (from structure.mjs) into per-module IR. A module is
// one `@module <name>` declaration — its output is `docs/api/<name>.md`.

import { callableReferenceKey, parseUtilityType, UTILITY_TYPES } from "./type-refs.mjs";

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

  const moduleList = [...modules.values()];
  resolveCallableAliases(moduleList);
  return { modules: moduleList, typedefIndex: buildTypedefIndex(modules) };
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
  const raw = fileTag?.description || entities.module.description || "";
  const { description, examples } = gatherExamples(raw);
  return { name: moduleTag.name, description, examples };
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
    const { description, examples } = gatherExamples(cls.description);
    mod.classes.push({
      name: cls.name,
      description,
      examples,
      skillExamples: tagsOf(cls, "skillExample").map((t) => t.task),
      members: cls.members.filter((m) => !isPrivate(m)).map(buildMember),
    });
  }
  for (const fn of entities.functions) {
    if (!isPrivate(fn)) mod.functions.push(buildCallable(fn));
  }
  for (const v of entities.variables) {
    if (isPrivate(v)) continue;
    // A `const` annotated with `@param` / `@returns` is a function alias —
    // hoist it into the functions section so it renders with a proper signature.
    if (tagsOf(v, "param").length || tagsOf(v, "returns").length) {
      mod.functions.push(buildCallable(v));
      continue;
    }
    const { description, examples } = gatherExamples(v.description);
    mod.constants.push({
      name: v.name,
      type: typeOf(v),
      description,
      examples,
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
  const { description, examples } = gatherExamples(fn.description);
  return {
    name: fn.name,
    description,
    params: tagsOf(fn, "param").map(normalizeParam),
    returns: pickReturns(fn),
    aliasType: typeOf(fn),
    throws: tagsOf(fn, "throws").map((t) => ({ type: t.type, description: t.description })),
    // `@template {Constraint} Name` maps the generic name to its constraint;
    // used by the renderer to resolve generic parameter names to something
    // readable instead of a bare `any`.
    templates: tagsOf(fn, "template")
      .map((t) => ({ name: t.name, type: t.type }))
      .filter((t) => t.name),
    examples,
    skillExamples: tagsOf(fn, "skillExample").map((t) => t.task),
    deprecated: fn.tags.some((t) => t.tag === "deprecated"),
  };
}

function resolveCallableAliases(modules) {
  const callableIndex = buildCallableIndex(modules);

  for (const callable of allCallables(modules)) {
    const target = findCallable(callable.aliasType, callableIndex);
    if (target && target !== callable) inheritCallable(callable, target);
  }

  for (const callable of allCallables(modules)) {
    for (const param of callable.params ?? []) {
      param.type = resolveUtilityType(param.type, callableIndex) ?? param.type;
    }
    if (callable.returns?.type) {
      callable.returns.type = resolveUtilityType(callable.returns.type, callableIndex) ?? callable.returns.type;
    }
  }
}

function buildCallableIndex(modules) {
  const index = new Map();
  for (const mod of modules) {
    for (const fn of mod.functions) index.set(fn.name, fn);
    for (const cls of mod.classes) {
      for (const m of cls.members) {
        if (m.kind === "method") index.set(`${cls.name}.${m.name}`, m);
      }
    }
  }
  return index;
}

function* allCallables(modules) {
  for (const mod of modules) {
    yield* mod.functions;
    for (const cls of mod.classes) {
      for (const m of cls.members) {
        if (m.kind === "method") yield m;
      }
    }
  }
}

function inheritCallable(callable, target) {
  if (!callable.description) callable.description = target.description;
  if (!callable.params?.length && target.params?.length) callable.params = clone(target.params);
  if (!callable.returns && target.returns) callable.returns = clone(target.returns);
  if (!callable.throws?.length && target.throws?.length) callable.throws = clone(target.throws);
  if (!callable.templates?.length && target.templates?.length) callable.templates = clone(target.templates);
  if (!callable.examples?.length && target.examples?.length) callable.examples = clone(target.examples);
}

function resolveUtilityType(type, callableIndex) {
  const utility = parseUtilityType(type);
  if (!utility) return null;

  const target = findCallable(utility.target, callableIndex);
  if (!target) return null;

  if (utility.kind === UTILITY_TYPES.PARAMETERS && utility.index != null) {
    return target.params?.[utility.index]?.type ?? null;
  }

  if (utility.kind === UTILITY_TYPES.RETURN_TYPE && utility.index == null) {
    return target.returns?.type ?? null;
  }

  return null;
}

function findCallable(ref, callableIndex) {
  const key = callableReferenceKey(ref);
  return key ? callableIndex.get(key) : null;
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
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

// Canonical example format: `**Example:** <title>\n```lang\n<code>\n```` inside
// any JSDoc description body. Extracted into a structured `examples` array so
// renderers can emit them consistently; the source lines are stripped from
// the description so they're not rendered twice.
const INLINE_EXAMPLE = /\*\*Example:\*\*\s*([^\n]*)\n+```(\w+)?\n([\s\S]*?)\n```/g;

function gatherExamples(description) {
  const examples = [];
  const cleaned = (description || "").replace(INLINE_EXAMPLE, (_, title, lang, code) => {
    examples.push({ title: title.trim(), language: lang || "javascript", code: code.trim() });
    return "";
  });
  return { description: cleaned.replace(/\n{3,}/g, "\n\n").trim(), examples };
}

function isPrivate(entity) {
  return entity.tags?.some((t) => t.tag === "private" || t.tag === "internal");
}
