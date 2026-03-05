/// <reference types="vite/client" />
// ForgeEngine — WebGPU Harness (self-testing)
import testWriteSrc from "./shaders/test_write.wgsl?raw";
import fillColorSrc from "./shaders/fill_color.wgsl?raw";

interface TestResult { name: string; pass: boolean; detail: string }

export async function initGPU(): Promise<{ adapter: GPUAdapter; device: GPUDevice }> {
  if (!navigator.gpu) throw new Error("WebGPU not supported in this browser");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPUAdapter found — no compatible GPU?");
  const device = await adapter.requestDevice();
  return { adapter, device };
}

function configureCanvas(device: GPUDevice, canvas: HTMLCanvasElement): GPUCanvasContext {
  const ctx = canvas.getContext("webgpu");
  if (!ctx) throw new Error("Failed to get WebGPU canvas context");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });
  return ctx;
}

export function createBuffer(device: GPUDevice, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size, usage });
}

export async function readbackBuffer(device: GPUDevice, src: GPUBuffer, size: number): Promise<Uint32Array> {
  const staging = createBuffer(device, size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

export function makeComputePipeline(device: GPUDevice, code: string) {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
}

export function dispatchCompute(device: GPUDevice, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: number) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([enc.finish()]);
}

// ── Test 1: GPU Init ──

async function testGPUInit(): Promise<TestResult & { device?: GPUDevice }> {
  try {
    const { adapter, device } = await initGPU();
    return { name: "GPU Init", pass: true, detail: `adapter: ${adapter.info?.vendor ?? "unknown"}, device OK`, device };
  } catch (e: any) {
    return { name: "GPU Init", pass: false, detail: e.message };
  }
}

// ── Test 2: Buffer Readback ──

async function testBufferReadback(device: GPUDevice): Promise<TestResult> {
  const COUNT = 256, BYTES = COUNT * 4;
  const buf = createBuffer(device, BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const pipeline = makeComputePipeline(device, testWriteSrc);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  });
  dispatchCompute(device, pipeline, bindGroup, Math.ceil(COUNT / 64));
  const data = await readbackBuffer(device, buf, BYTES);
  buf.destroy();

  let failIdx = -1;
  for (let i = 0; i < COUNT; i++) { if (data[i] !== i) { failIdx = i; break; } }
  if (failIdx === -1) return { name: "Buffer Readback", pass: true, detail: `${COUNT}/${COUNT} values correct` };
  return { name: "Buffer Readback", pass: false, detail: `FAIL at index ${failIdx}: expected ${failIdx}, got ${data[failIdx]}` };
}

// ── Test 3: Framebuffer Fill ──

async function testFramebufferFill(
  device: GPUDevice, width: number, height: number
): Promise<TestResult & { framebuffer?: GPUBuffer }> {
  const PIXELS = width * height, BYTES = PIXELS * 4;
  const COLOR = 0xff6495ed; // arbitrary test value written by compute, verified on readback

  const fbBuf = createBuffer(device, BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const paramData = new ArrayBuffer(16);
  const v = new DataView(paramData);
  v.setUint32(0, width, true); v.setUint32(4, height, true); v.setUint32(8, COLOR, true);
  const paramBuf = createBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(paramBuf, 0, paramData);

  const pipeline = makeComputePipeline(device, fillColorSrc);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fbBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
    ],
  });
  dispatchCompute(device, pipeline, bindGroup, Math.ceil(PIXELS / 64));

  const data = await readbackBuffer(device, fbBuf, BYTES);
  paramBuf.destroy();

  const spots = [0, Math.floor(PIXELS / 2), PIXELS - 1];
  const fails = spots.filter(i => data[i] !== COLOR);
  if (fails.length === 0) {
    return { name: "Framebuffer Fill", pass: true, detail: `${spots.length} spot checks passed`, framebuffer: fbBuf };
  }
  fbBuf.destroy();
  const info = fails.map(i => `pixel[${i}]: got 0x${data[i].toString(16)}`).join("; ");
  return { name: "Framebuffer Fill", pass: false, detail: info };
}

// ── Display ──

function blitToCanvas(device: GPUDevice, ctx: GPUCanvasContext, _fbBuf: GPUBuffer, _w: number, _h: number) {
  // Cornflower blue: R=0x64 G=0x95 B=0xED → normalized
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0x64 / 255, g: 0x95 / 255, b: 0xED / 255, a: 1 },
      loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp,
    }],
  });
  pass.end();
  device.queue.submit([enc.finish()]);
}

function blitSolidRed(device: GPUDevice, ctx: GPUCanvasContext) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 1, g: 0, b: 0, a: 1 },
      loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp,
    }],
  });
  pass.end();
  device.queue.submit([enc.finish()]);
}

// ── Main ──

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const results: TestResult[] = [];
  const log = (r: TestResult) => {
    results.push(r);
    console.log(`[${r.pass ? "✓ PASS" : "✗ FAIL"}] ${r.name}: ${r.detail}`);
  };

  const initResult = await testGPUInit();
  log(initResult);
  if (!initResult.pass || !initResult.device) { console.error("Cannot continue without GPU"); return; }
  const { device } = initResult;
  const ctx = configureCanvas(device, canvas);

  log(await testBufferReadback(device));

  const fbResult = await testFramebufferFill(device, canvas.width, canvas.height);
  log(fbResult);

  const allPass = results.every(r => r.pass);
  if (allPass && fbResult.framebuffer) {
    blitToCanvas(device, ctx, fbResult.framebuffer, canvas.width, canvas.height);
    fbResult.framebuffer.destroy();
  } else {
    blitSolidRed(device, ctx);
  }

  console.log(`\n── Harness: ${results.filter(r => r.pass).length}/${results.length} passed ──`);
}

// Only run self-tests when loaded as the entry point (not when imported)
if (document.getElementById("gpu-canvas")) {
  main();
}
