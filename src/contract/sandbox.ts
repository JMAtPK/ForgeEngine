import type { ResolvedSchema } from "../schema/types";
import type {
  ElementAccessor,
  KernelContractJSON,
  PostconditionContext,
  PostconditionResult,
  ResolvedContract,
  ResolvedBinding,
} from "./types";
import { readElement } from "../schema/schema";
import { eq, createFailFn, clamp, abs, min, max, fix16_mul } from "./helpers";

export function makeElementAccessor(buffer: ArrayBuffer, schema: ResolvedSchema): ElementAccessor {
  return new Proxy([] as any, {
    get(_target, prop) {
      if (prop === "length") return schema.capacity;
      if (typeof prop === "string") {
        const idx = Number(prop);
        if (!isNaN(idx) && idx >= 0 && idx < schema.capacity) {
          return readElement(buffer, schema, idx);
        }
      }
      if (prop === Symbol.iterator) {
        return function* () {
          for (let i = 0; i < schema.capacity; i++) {
            yield readElement(buffer, schema, i);
          }
        };
      }
      return undefined;
    },
  });
}

export function compilePostcondition(source: string): Function {
  return new Function(
    "input",
    "output",
    "globals",
    "$",
    "eq",
    "fail",
    "clamp",
    "abs",
    "min",
    "max",
    "fix16_mul",
    source
  );
}

export function evalPostcondition(
  compiledFn: Function,
  ctx: PostconditionContext
): PostconditionResult {
  const { fail, errors } = createFailFn();
  try {
    const result = compiledFn(
      ctx.input,
      ctx.output,
      ctx.globals,
      ctx.$,
      eq,
      fail,
      clamp,
      abs,
      min,
      max,
      fix16_mul
    );
    const pass = result !== false && errors.length === 0;
    return { pass, errors };
  } catch (e) {
    errors.push(`Postcondition threw: ${(e as Error).message}`);
    return { pass: false, errors };
  }
}

export function resolveContract(
  json: KernelContractJSON,
  schemas: Map<string, ResolvedSchema>
): ResolvedContract {
  const inputs: ResolvedBinding[] = json.inputs.map((b) => {
    const schema = schemas.get(b.schema);
    if (!schema) throw new Error(`Unknown schema: ${b.schema}`);
    return { binding: b.binding, schema, access: b.access };
  });

  const outputs: ResolvedBinding[] = json.outputs.map((b) => {
    const schema = schemas.get(b.schema);
    if (!schema) throw new Error(`Unknown schema: ${b.schema}`);
    return { binding: b.binding, schema, access: b.access };
  });

  const postconditionSources = json.postconditions;
  const postconditionFns = postconditionSources.map(compilePostcondition);

  return {
    name: json.name,
    inputs,
    outputs,
    postconditionSources,
    postconditionFns,
  };
}
