const IDENTIFIER = String.raw`[A-Za-z_$][\w$]*`;

export const UTILITY_TYPES = Object.freeze({
  PARAMETERS: "Parameters",
  RETURN_TYPE: "ReturnType",
  CONSTRUCTOR_PARAMETERS: "ConstructorParameters",
  INSTANCE_TYPE: "InstanceType",
  THIS_TYPE: "ThisType",
});

const RENDERABLE_UTILITY_NAMES = [UTILITY_TYPES.PARAMETERS, UTILITY_TYPES.RETURN_TYPE, UTILITY_TYPES.CONSTRUCTOR_PARAMETERS];
const RENDERABLE_UTILITY_PATTERN = new RegExp(`^(${RENDERABLE_UTILITY_NAMES.join("|")})<(.+)>(?:\\[(\\d+)\\])?$`);

export const TS_UTILITY_NAMES = new Set(Object.values(UTILITY_TYPES));

const CALLABLE_REF_PATTERNS = [
  {
    regex: new RegExp(`^typeof\\s+(${IDENTIFIER})\\.(${IDENTIFIER})$`),
    parse: (m) => ({ owner: m[1], method: m[2] }),
  },
  {
    regex: new RegExp(`^typeof\\s+(${IDENTIFIER})$`),
    parse: (m) => ({ owner: m[1], method: null }),
  },
  {
    regex: new RegExp(`^(${IDENTIFIER})\\[['"]([^'"]+)['"]\\]$`),
    parse: (m) => ({ owner: m[1], method: m[2] }),
  },
  {
    regex: new RegExp(`^(${IDENTIFIER})\\.(${IDENTIFIER})$`),
    parse: (m) => ({ owner: m[1], method: m[2] }),
  },
];

export function parseUtilityType(raw) {
  if (!raw) return null;
  const match = raw.trim().match(RENDERABLE_UTILITY_PATTERN);
  return match
    ? {
        kind: match[1],
        target: match[2],
        index: match[3] == null ? null : Number(match[3]),
      }
    : null;
}

export function isRenderableUtilityType(raw) {
  return parseUtilityType(raw) !== null;
}

export function parseCallableReference(raw) {
  if (!raw) return null;
  const text = raw.trim();
  for (const { regex, parse } of CALLABLE_REF_PATTERNS) {
    const match = text.match(regex);
    if (match) return parse(match);
  }
  return null;
}

export function callableReferenceKey(raw) {
  const ref = parseCallableReference(raw);
  if (!ref) return null;
  return ref.method ? `${ref.owner}.${ref.method}` : ref.owner;
}
