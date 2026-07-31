// Checks if type X and Y are exactly the same
type Equal<X, Y> = 
  (<T>() => T extends X ? 1 : 2) extends 
  (<T>() => T extends Y ? 1 : 2) ? true : false;

// Throws a type error if T is not `true`
type Expect<T extends true> = T;

// Throws a type error if T is not `false`
type ExpectError<T extends false> = T;

export type { Expect, Equal, ExpectError };
