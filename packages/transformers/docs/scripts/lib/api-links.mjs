export const DOCS_BASE_URL = "https://huggingface.co/docs/transformers.js";

export function buildApiSymbolLinks(ir, publicNames, baseUrl = DOCS_BASE_URL) {
  const links = new Map();
  const ambiguous = new Set();
  for (const mod of ir.modules) {
    for (const cls of mod.classes) {
      if (!isPublic(cls, publicNames)) continue;
      addLink(links, ambiguous, cls.name, apiUrl(mod.name, cls.name, baseUrl));
    }
    for (const fn of mod.functions) {
      if (!isPublic(fn, publicNames)) continue;
      addLink(links, ambiguous, fn.name, apiUrl(mod.name, fn.name, baseUrl));
    }
    for (const constant of mod.constants) {
      if (!isPublic(constant, publicNames)) continue;
      addLink(links, ambiguous, constant.name, apiUrl(mod.name, constant.name, baseUrl));
    }
    for (const td of mod.typedefs) addLink(links, ambiguous, td.name, apiUrl(mod.name, td.name, baseUrl));
    for (const cb of mod.callbacks) addLink(links, ambiguous, cb.name, apiUrl(mod.name, cb.name, baseUrl));
  }
  return links;
}

export function apiUrl(moduleName, symbolName, baseUrl = DOCS_BASE_URL) {
  return `${baseUrl}/api/${moduleName}#${apiSymbolAnchor(moduleName, symbolName)}`;
}

export function apiSymbolAnchor(moduleName, symbolName) {
  return `module_${moduleName}.${symbolName}`;
}

export function apiMemberAnchor(moduleName, parentName, memberName) {
  return apiSymbolAnchor(moduleName, `${parentName}.${memberName}`);
}

function isPublic(item, publicNames) {
  return !publicNames || publicNames.has(item.name);
}

function addLink(links, ambiguous, name, link) {
  if (ambiguous.has(name)) return;

  const previous = links.get(name);
  if (previous && previous !== link) {
    links.delete(name);
    ambiguous.add(name);
    return;
  }

  links.set(name, link);
}
