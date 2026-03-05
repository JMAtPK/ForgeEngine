import { fix16Mul } from "../schema/params";

export function eq(a: Record<string, number>, b: Record<string, number>): boolean {
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const key of keysA) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

export function createFailFn(): { fail: (index: number, msg: string) => false; errors: string[] } {
  const errors: string[] = [];
  const fail = (index: number, msg: string): false => {
    errors.push(`[${index}] ${msg}`);
    return false;
  };
  return { fail, errors };
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function abs(v: number): number {
  return v < 0 ? -v : v;
}

export function min(a: number, b: number): number {
  return a < b ? a : b;
}

export function max(a: number, b: number): number {
  return a > b ? a : b;
}

export const fix16_mul = fix16Mul;
