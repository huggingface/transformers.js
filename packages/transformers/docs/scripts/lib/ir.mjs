// Group per-file entities (from structure.mjs) into per-module IR. A module is
// one `@module <name>` declaration — its output is `docs/api/<name>.md`.

import path from "node:path";

import { findTopLevel, matchingBracket, splitTopLevel } from "./scan.mjs";
import { parseImportedType, stripImportPrefixes } from "./text.mjs";
import { parseCallableReference, parseUtilityType, UTILITY_TYPES } from "./type-refs.mjs";

export function buildIR(fileEntities) {
  const modules = new Map();
  const records = fileEntities.map(({ file, entities }) => ({ file, entities, module: resolveModule(entities) }));
  const moduleByFile = new Map(records.filter((r) => r.module).map((r) => [r.file, r.module.name]));
  const typeBlocksByFile = buildTypeBlocksByFile(records);

  for (const { file, entities, module: mod } of records) {
    if (!mod) continue;
    const entry = modules.get(mod.name) ?? newModule(mod.name);
    if (!entry.description && mod.description) entry.description = mod.description;
    entry.examples.push(...mod.examples);
    ingest(entities, entry);
    for (const helperFile of importedHelperFiles(entities, file, moduleByFile, typeBlocksByFile)) {
      entry.helperFiles.add(helperFile);
    }
    modules.set(mod.name, entry);
  }

  const moduleList = [...modules.values()];
  attachImportedHelperTypes(moduleList, typeBlocksByFile);
  attachClassCallables(moduleList);
  resolveCallableAliases(moduleList);
  return { modules: moduleList, typedefIndex: buildTypedefIndex(modules) };
}

// Classes are authoritative for a name; typedefs fill gaps. Re-export aliases
// (`@typedef {import('x').Foo} Foo`) register last so they can't shadow the canonical definition.
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
  return stripImportPrefixes(td.type).trim() === td.name;
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
    helperFiles: new Set(),
  };
}

function buildTypeBlocksByFile(records) {
  const byFile = new Map();
  for (const { file, entities } of records) {
    const bucket = { typedefs: [], callbacks: [] };
    for (const block of entities.typedefs) {
      if (!isPrivate(block)) collectTypedefsFromBlock(block, bucket);
    }
    byFile.set(file, bucket);
  }
  return byFile;
}

function importedHelperFiles(entities, file, moduleByFile, typeBlocksByFile) {
  const out = new Set();
  for (const block of entities.typedefs) {
    for (const tag of block.tags) {
      if (tag.tag !== "typedef") continue;
      const ref = parseImportedTypeRef(tag.type, file);
      if (!ref || moduleByFile.has(ref.file) || !typeBlocksByFile.has(ref.file)) continue;
      out.add(ref.file);
    }
  }
  return out;
}

function attachImportedHelperTypes(modules, typeBlocksByFile) {
  for (const mod of modules) {
    if (!mod.helperFiles.size) continue;
    const helpers = { typedefs: [], callbacks: [] };
    for (const file of [...mod.helperFiles].sort()) {
      const block = typeBlocksByFile.get(file);
      if (!block) continue;
      mergeByName(helpers.typedefs, block.typedefs);
      mergeByName(helpers.callbacks, block.callbacks);
    }
    mod.typedefs = dedupeTypes([...helpers.typedefs, ...mod.typedefs]);
    mod.callbacks = dedupeTypes([...helpers.callbacks, ...mod.callbacks]);
  }
}

function mergeByName(target, items) {
  const seen = new Set(target.map((item) => item.name));
  for (const item of items) {
    if (!item.name || seen.has(item.name)) continue;
    target.push(structuredClone(item));
    seen.add(item.name);
  }
}

// One entry per name, at its first occurrence. When both a re-export alias
// (`@typedef {import('./x.js').Foo} Foo`) and the real declaration exist, the real one wins:
// otherwise the name renders as a self-reference and every link to it dangles.
function dedupeTypes(items) {
  const best = new Map();
  for (const item of items) {
    if (!item.name) continue;
    const existing = best.get(item.name);
    if (!existing || (isAliasTypedef(existing) && !isAliasTypedef(item))) best.set(item.name, item);
  }
  const emitted = new Set();
  return items.flatMap((item) => {
    if (!item.name || emitted.has(item.name)) return [];
    emitted.add(item.name);
    return [best.get(item.name)];
  });
}

function parseImportedTypeRef(type, fromFile) {
  const ref = parseImportedType(type);
  if (!ref || !ref.specifier.startsWith(".")) return null;
  return { file: resolveJsPath(path.resolve(path.dirname(fromFile), ref.specifier)), name: ref.name };
}

function resolveJsPath(file) {
  return path.extname(file) ? file : `${file}.js`;
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
      see: seeTagsOf(cls),
      members: cls.members.filter((m) => !isPrivate(m)).map(buildMember),
      callable: null,
    });
  }
  for (const fn of entities.functions) {
    if (!isPrivate(fn)) mod.functions.push(buildCallable(fn));
  }
  for (const v of entities.variables) {
    if (isPrivate(v)) continue;
    // A `const` annotated with `@param` / `@returns` is a function alias; hoist it into the
    // functions section so it renders with a signature.
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
      see: seeTagsOf(v),
    });
  }
}

// One `@typedef`/`@callback` block may declare several types; `@property` tags attach to the
// preceding typedef, up to the next one.
function collectTypedefsFromBlock(block, mod) {
  let current = null;
  let first = true;
  // `@template` tags sit on the block, not on the individual `@typedef`. Carry them onto every type
  // the block declares so the renderer can resolve `T` instead of falling back to `any`.
  const blockTemplates = templatesOf(block);
  const flush = () => {
    if (!current) return;
    if (current.kind === "typedef") mod.typedefs.push(current);
    else mod.callbacks.push(current);
    current = null;
  };
  for (const tag of block.tags) {
    if (tag.tag === "typedef") {
      flush();
      const callable = parseCallableTypedef({
        name: tag.name,
        type: tag.type,
        description: tag.description || (first ? block.description : ""),
      });
      current = callable
        ? { ...callable, kind: "callback", templates: callable.templates?.length ? callable.templates : blockTemplates }
        : {
            kind: "typedef",
            name: tag.name,
            type: tag.type,
            description: tag.description || (first ? block.description : ""),
            templates: blockTemplates,
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
        templates: blockTemplates,
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
    // A getter may document its type as `@returns {T}` rather than `@type`; without the fallback
    // the member has no type at all.
    type: typeOf(m) ?? (m.kind === "getter" ? (pickReturns(m)?.type ?? null) : null),
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
    see: seeTagsOf(fn),
    throws: tagsOf(fn, "throws").map((t) => ({ type: t.type, description: t.description })),
    // `@template {Constraint} Name` maps a generic name to its constraint, which the renderer shows
    // instead of a bare `any`.
    templates: templatesOf(fn),
    examples,
    deprecated: fn.tags.some((t) => t.tag === "deprecated"),
  };
}

function attachClassCallables(modules) {
  for (const mod of modules) {
    const callbacks = new Map(mod.callbacks.map((cb) => [cb.name, cb]));

    for (const cls of mod.classes) {
      const callMember = cls.members.find((m) => m.kind === "method" && m.name === "_call");
      const callbackName = `${cls.name}Callback`;
      const callback = callbacks.get(callbackName);
      const callSource = callMember && hasCallableShape(callMember) ? callMember : null;
      // The `XCallback` typedef has the precise generic signature but no prose; a documented
      // `_call` has both. Prefer whichever explains what calling the instance does.
      const source = (callSource?.description ? callSource : null) ?? callback ?? callSource;
      if (!source) continue;

      // Promote the source description onto the synthesized signature; without it the page shows a
      // bare `Foo(...)` line.
      cls.callable = {
        ...normalizeCallable(source),
        name: cls.name,
        description: source.description ?? "",
        deprecated: false,
      };
    }
  }
}

function normalizeCallable(callable) {
  const cloned = structuredClone(callable);
  return {
    ...cloned,
    params: cloned.params ?? [],
    returns: cloned.returns ?? null,
    aliasType: cloned.aliasType ?? null,
    throws: cloned.throws ?? [],
    templates: cloned.templates ?? [],
    examples: cloned.examples ?? [],
  };
}

function hasCallableShape(callable) {
  return !!(callable.params?.length || callable.returns?.type || callable.returns?.description || callable.aliasType);
}

function parseCallableTypedef(td) {
  const parsed = parseFunctionType(td.type);
  if (!parsed) return null;
  return {
    name: td.name,
    description: td.description,
    params: parsed.params,
    returns: parsed.returns,
    aliasType: null,
    throws: [],
    templates: parsed.templates,
    examples: [],
    deprecated: false,
  };
}

function parseFunctionType(raw) {
  if (!raw) return null;
  let text = raw.trim();
  const templates = [];

  if (text.startsWith("<")) {
    const end = matchingBracket(text, 0, "<", ">");
    if (end === -1) return null;
    templates.push(...parseTemplateList(text.slice(1, end)));
    text = text.slice(end + 1).trim();
  }

  if (!text.startsWith("(")) return null;
  const paramsEnd = matchingBracket(text, 0, "(", ")");
  if (
    paramsEnd === -1 ||
    text
      .slice(paramsEnd + 1)
      .trimStart()
      .slice(0, 2) !== "=>"
  )
    return null;

  const params = splitTopLevel(text.slice(1, paramsEnd), ",")
    .map((part) => parseFunctionTypeParam(part.trim()))
    .filter(Boolean);
  const returnType = text
    .slice(paramsEnd + 1)
    .trimStart()
    .slice(2)
    .trim();

  return {
    templates,
    params,
    returns: returnType ? { type: returnType, description: "" } : null,
  };
}

// A template parameter may carry a `const` modifier and a default
// (`<Q extends string | string[], const O extends Options = {}>`). Both are dropped, but the entry
// still has to parse, or its name renders unresolved.
function parseTemplateList(raw) {
  return splitTopLevel(raw, ",")
    .map((part) => {
      let text = part.trim().replace(/^const\s+/, "");
      const eq = findTopLevel(text, "=");
      // `=` starts a default; `=>` is a function-type arrow inside a constraint.
      if (eq !== -1 && text[eq + 1] !== ">") text = text.slice(0, eq).trim();
      const m = text.match(/^([A-Za-z_$][\w$]*)(?:\s+extends\s+(.+))?$/);
      return m ? { name: m[1], type: m[2]?.trim() ?? null } : null;
    })
    .filter(Boolean);
}

function parseFunctionTypeParam(raw) {
  if (!raw) return null;
  const colon = findTopLevel(raw, ":");
  if (colon === -1) return { name: raw.replace(/\?$/, ""), type: null, optional: raw.endsWith("?"), defaultValue: null, description: "" };
  const name = raw.slice(0, colon).trim();
  return {
    name: name.replace(/\?$/, ""),
    type: raw.slice(colon + 1).trim(),
    optional: name.endsWith("?"),
    defaultValue: null,
    description: "",
  };
}

function resolveCallableAliases(modules) {
  const callableIndex = buildCallableIndex(modules);

  for (const callable of allCallables(modules)) {
    const target = findCallable(callable.aliasType, callableIndex);
    if (target && target !== callable) inheritCallable(callable, target);
  }

  for (const callable of allCallables(modules)) {
    callable.params = resolveUtilityParams(callable.params ?? [], callableIndex);
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
      if (cls.callable) index.set(cls.name, cls.callable);
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
      if (cls.callable) yield cls.callable;
      for (const m of cls.members) {
        if (m.kind === "method") yield m;
      }
    }
  }
}

function inheritCallable(callable, target) {
  if (!callable.description) callable.description = target.description;
  if (!callable.params?.length && target.params?.length) callable.params = structuredClone(target.params);
  if (!callable.returns && target.returns) callable.returns = structuredClone(target.returns);
  if (!callable.throws?.length && target.throws?.length) callable.throws = structuredClone(target.throws);
  if (!callable.templates?.length && target.templates?.length) callable.templates = structuredClone(target.templates);
  if (!callable.examples?.length && target.examples?.length) callable.examples = structuredClone(target.examples);
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

function resolveUtilityParams(params, callableIndex) {
  const resolved = [];
  for (const param of params) {
    const utility = parseUtilityType(param.type);
    const target = utility && findCallable(utility.target, callableIndex);
    if (utility?.kind === UTILITY_TYPES.PARAMETERS && utility.index == null && target?.params?.length) {
      resolved.push(...structuredClone(target.params));
      continue;
    }
    resolved.push({
      ...param,
      type: resolveUtilityType(param.type, callableIndex) ?? param.type,
    });
  }
  return resolved;
}

function findCallable(ref, callableIndex) {
  const parsed = parseCallableReference(ref);
  if (!parsed) return null;
  return callableIndex.get(parsed.method ? `${parsed.owner}.${parsed.method}` : parsed.owner);
}

// `@template {Constraint} Name` -> `{ name, type }`, the shape the renderer resolves against.
function templatesOf(entity) {
  return tagsOf(entity, "template")
    .map((t) => ({ name: t.name, type: t.type }))
    .filter((t) => t.name);
}

function tagsOf(entity, name) {
  return entity.tags.filter((t) => t.tag === name);
}

function seeTagsOf(entity) {
  return tagsOf(entity, "see")
    .map((t) => t.description)
    .filter(Boolean);
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
    description: cleanParamDescription(tag.description || ""),
  };
}

function cleanParamDescription(description) {
  return description.replace(/^\([^)]*\boptional\b[^)]*\):\s*/i, "").trim();
}

// Canonical example format inside any JSDoc description body. Matches are moved into a structured
// `examples` array and stripped from the description so they aren't rendered twice.
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
