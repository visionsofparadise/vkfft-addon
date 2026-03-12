const path = require("path");
const addon = require(path.join(__dirname, "build/Release/vkfft_addon.node"));

function maxError(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`PASS: ${name}`);
  } catch (e) {
    console.log(`FAIL: ${name} — ${e.message}`);
    failed++;
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// detectDevice — CI runners have no GPU, so null is expected
test("detectDevice returns string or null", () => {
  const device = addon.detectDevice();
  assert(device === null || typeof device === "string", `unexpected type: ${typeof device}`);
  console.log(`  device: ${device}`);
});

const hasGpu = addon.detectDevice() !== null;

if (hasGpu) {
  test("batchFft returns correct shape", () => {
    const fftSize = 1024;
    const input = new Float32Array(fftSize).fill(0);
    input[0] = 1;
    const { re, im } = addon.batchFft(input, fftSize, 1);
    assert(re.length === fftSize / 2 + 1, `re length ${re.length}`);
    assert(im.length === fftSize / 2 + 1, `im length ${im.length}`);
    assert(Math.abs(re[0] - 1) < 1e-4, `DC bin re=${re[0]}`);
    assert(Math.abs(im[0]) < 1e-4, `DC bin im=${im[0]}`);
  });

  test("round-trip fft -> ifft", () => {
    const fftSize = 1024;
    const batchCount = 4;
    const input = new Float32Array(fftSize * batchCount);
    for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
    const { re, im } = addon.batchFft(input, fftSize, batchCount);
    const output = addon.batchIfft(re, im, fftSize, batchCount);
    const err = maxError(input, output);
    assert(err < 1e-4, `error ${err}`);
    console.log(`  max error: ${err.toExponential(2)}`);
  });

  test("multiple FFT sizes", () => {
    for (const fftSize of [256, 512, 2048, 4096]) {
      const input = new Float32Array(fftSize);
      for (let i = 0; i < fftSize; i++) input[i] = Math.sin(2 * Math.PI * i / fftSize);
      const { re, im } = addon.batchFft(input, fftSize, 1);
      const output = addon.batchIfft(re, im, fftSize, 1);
      const err = maxError(input, output);
      assert(err < 1e-4, `fftSize=${fftSize} error=${err}`);
    }
  });

  test("large batch", () => {
    const fftSize = 1024;
    const batchCount = 100;
    const input = new Float32Array(fftSize * batchCount);
    for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
    const { re, im } = addon.batchFft(input, fftSize, batchCount);
    const output = addon.batchIfft(re, im, fftSize, batchCount);
    const err = maxError(input, output);
    assert(err < 1e-4, `error ${err}`);
    console.log(`  max error: ${err.toExponential(2)}`);
  });
} else {
  console.log("No GPU detected — skipping FFT tests (CI expected)");
}

process.exit(failed);
