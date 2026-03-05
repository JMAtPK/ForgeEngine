import type { ResolvedParams } from "../schema/types";
import type {
  ResolvedContract,
  CheckResult,
  PostconditionResult,
  InvariantResult,
  CleanWriteResult,
} from "./types";
import { readElement, validateBuffer } from "../schema/schema";
import {
  createBuffer,
  readbackBuffer,
  dispatchCompute,
} from "../harness";
import { makeElementAccessor, evalPostcondition } from "./sandbox";

export function snapshotBuffer(buffer: ArrayBuffer): ArrayBuffer {
  return buffer.slice(0);
}

export function checkCleanWrite(
  contract: ResolvedContract,
  preSnapshots: ArrayBuffer[],
  postBuffers: ArrayBuffer[]
): CleanWriteResult {
  const violations: string[] = [];

  for (let i = 0; i < contract.inputs.length; i++) {
    if (contract.inputs[i].access !== "read") continue;

    const pre = new Uint8Array(preSnapshots[i]);
    const post = new Uint8Array(postBuffers[i]);

    if (pre.length !== post.length) {
      violations.push(`Input binding ${contract.inputs[i].binding}: size changed (${pre.length} → ${post.length})`);
      continue;
    }

    for (let j = 0; j < pre.length; j++) {
      if (pre[j] !== post[j]) {
        violations.push(
          `Input binding ${contract.inputs[i].binding}: modified at byte ${j} (was ${pre[j]}, now ${post[j]})`
        );
        break;
      }
    }
  }

  return { pass: violations.length === 0, violations };
}

export function checkSchemaInvariants(
  contract: ResolvedContract,
  outputBuffers: ArrayBuffer[],
  params: ResolvedParams
): InvariantResult[] {
  const results: InvariantResult[] = [];

  for (let i = 0; i < contract.outputs.length; i++) {
    const schema = contract.outputs[i].schema;
    const { valid, errors } = validateBuffer(outputBuffers[i], schema, params);
    results.push({ schema: schema.name, pass: valid, errors });
  }

  return results;
}

export function checkPostconditions(
  contract: ResolvedContract,
  inputSnapshots: ArrayBuffer[],
  outputBuffers: ArrayBuffer[],
  params: ResolvedParams
): PostconditionResult[] {
  const inputSchema = contract.inputs[0].schema;
  const outputSchema = contract.outputs[0].schema;

  const input = makeElementAccessor(inputSnapshots[0], inputSchema);
  const output = makeElementAccessor(outputBuffers[0], outputSchema);

  // Find globals binding
  let globals: Record<string, number> = {};
  for (let i = 0; i < contract.inputs.length; i++) {
    if (contract.inputs[i].schema.name.includes("Globals")) {
      globals = readElement(inputSnapshots[i], contract.inputs[i].schema, 0);
      break;
    }
  }

  const $ = params.raw;

  return contract.postconditionFns.map((fn) =>
    evalPostcondition(fn, { input, output, globals, $ })
  );
}

export async function checkContractGPU(
  device: GPUDevice,
  contract: ResolvedContract,
  wgslSource: string,
  inputBuffers: ArrayBuffer[],
  params: ResolvedParams
): Promise<CheckResult> {
  // 1. Compile WGSL → compute pipeline
  const shaderModule = device.createShaderModule({ code: wgslSource });
  const compilationInfo = await shaderModule.getCompilationInfo();
  for (const msg of compilationInfo.messages) {
    if (msg.type === "error") {
      throw new Error(`WGSL compilation error: ${msg.message} (line ${msg.lineNum})`);
    }
  }
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });

  // 2. Create GPU buffers
  const gpuInputBuffers: GPUBuffer[] = [];
  const gpuOutputBuffers: GPUBuffer[] = [];

  for (let i = 0; i < contract.inputs.length; i++) {
    const size = contract.inputs[i].schema.totalSize;
    const buf = createBuffer(
      device,
      size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    // 3. Upload input data
    device.queue.writeBuffer(buf, 0, inputBuffers[i]);
    gpuInputBuffers.push(buf);
  }

  for (let i = 0; i < contract.outputs.length; i++) {
    const size = contract.outputs[i].schema.totalSize;
    const buf = createBuffer(
      device,
      size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    gpuOutputBuffers.push(buf);
  }

  // 4. Snapshot inputs before dispatch
  const inputSnapshots = inputBuffers.map(snapshotBuffer);

  // 5. Create bind group and dispatch
  device.pushErrorScope("validation");
  const entries: GPUBindGroupEntry[] = [];
  for (let i = 0; i < contract.inputs.length; i++) {
    entries.push({
      binding: contract.inputs[i].binding,
      resource: { buffer: gpuInputBuffers[i] },
    });
  }
  for (let i = 0; i < contract.outputs.length; i++) {
    entries.push({
      binding: contract.outputs[i].binding,
      resource: { buffer: gpuOutputBuffers[i] },
    });
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const capacity = contract.inputs[0].schema.capacity;
  const workgroups = Math.ceil(capacity / 64);
  dispatchCompute(device, pipeline, bindGroup, workgroups);
  const gpuError = await device.popErrorScope();
  if (gpuError) {
    throw new Error(`GPU validation error: ${gpuError.message}`);
  }
  await device.queue.onSubmittedWorkDone();

  // 6. Read back all buffers
  const postInputBuffers: ArrayBuffer[] = [];
  for (const buf of gpuInputBuffers) {
    const data = await readbackBuffer(device, buf, buf.size);
    postInputBuffers.push(data.buffer as ArrayBuffer);
  }

  const postOutputBuffers: ArrayBuffer[] = [];
  for (const buf of gpuOutputBuffers) {
    const data = await readbackBuffer(device, buf, buf.size);
    postOutputBuffers.push(data.buffer as ArrayBuffer);
  }

  // Cleanup GPU buffers
  gpuInputBuffers.forEach((b) => b.destroy());
  gpuOutputBuffers.forEach((b) => b.destroy());

  // 7. Run all checks
  const cleanWrite = checkCleanWrite(contract, inputSnapshots, postInputBuffers);
  const invariants = checkSchemaInvariants(contract, postOutputBuffers, params);
  const postconditions = checkPostconditions(
    contract,
    inputSnapshots,
    postOutputBuffers,
    params
  );

  const pass =
    cleanWrite.pass &&
    invariants.every((inv) => inv.pass) &&
    postconditions.every((pc) => pc.pass);

  return { pass, postconditions, invariants, cleanWrite };
}
