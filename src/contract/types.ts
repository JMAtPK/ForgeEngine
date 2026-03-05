import type { ResolvedSchema } from "../schema/types";

// ── Contract JSON (as declared by kernel author) ──

export interface ContractBinding {
  binding: number;
  schema: string;
  access: "read" | "write";
}

export interface KernelContractJSON {
  name: string;
  description: string;
  inputs: ContractBinding[];
  outputs: ContractBinding[];
  postconditions: string[];
  adversarial_cases?: string[];
}

// ── Resolved (ready for execution) ──

export interface ResolvedBinding {
  binding: number;
  schema: ResolvedSchema;
  access: "read" | "write";
}

export interface ResolvedContract {
  name: string;
  inputs: ResolvedBinding[];
  outputs: ResolvedBinding[];
  postconditionSources: string[];
  postconditionFns: Function[];
}

// ── Postcondition evaluation context ──

export interface ElementAccessor {
  readonly length: number;
  [index: number]: Record<string, number>;
}

export interface PostconditionContext {
  input: ElementAccessor;
  output: ElementAccessor;
  globals: Record<string, number>;
  $: Record<string, number>;
}

// ── Check results ──

export interface PostconditionResult {
  pass: boolean;
  errors: string[];
}

export interface InvariantResult {
  schema: string;
  pass: boolean;
  errors: string[];
}

export interface CleanWriteResult {
  pass: boolean;
  violations: string[];
}

export interface CheckResult {
  pass: boolean;
  postconditions: PostconditionResult[];
  invariants: InvariantResult[];
  cleanWrite: CleanWriteResult;
}
