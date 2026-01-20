// Set up the React act() environment for testing
// This needs to be set on the React internals
if (typeof globalThis !== "undefined") {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
}

if (typeof global !== "undefined") {
  global.IS_REACT_ACT_ENVIRONMENT = true;
}
