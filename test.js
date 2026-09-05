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

function createRandomSignal(length) {
	const signal = new Float32Array(length);
	for (let i = 0; i < length; i++) signal[i] = Math.random() * 2 - 1;
	return signal;
}

function assertRoundTrip(input, output, tolerance) {
	const error = maxError(input, output);
	assert(error < tolerance, `error ${error}`);
	console.log(`  max error: ${error.toExponential(2)}`);
}

function assertBatchRoundTrip(fftSize, batchCount, tolerance) {
	const input = createRandomSignal(fftSize * batchCount);
	const { re, im } = addon.batchFft(input, fftSize, batchCount);
	const output = addon.batchIfft(re, im, fftSize, batchCount);
	assertRoundTrip(input, output, tolerance);
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

	test("round-trip fft -> ifft", () => assertBatchRoundTrip(1024, 4, 1e-4));

	test("multiple FFT sizes", () => {
		for (const fftSize of [256, 512, 2048, 4096]) {
			const input = new Float32Array(fftSize);
			for (let i = 0; i < fftSize; i++) input[i] = Math.sin((2 * Math.PI * i) / fftSize);
			const { re, im } = addon.batchFft(input, fftSize, 1);
			const output = addon.batchIfft(re, im, fftSize, 1);
			const error = maxError(input, output);
			assert(error < 1e-4, `fftSize=${fftSize} error=${error}`);
		}
	});

	test("large batch", () => assertBatchRoundTrip(1024, 100, 1e-4));

	// ── *Into variants (caller-owned output buffers) ─────────────────────────

	test("batchFftInto / batchIfftInto round-trip", () => {
		const fftSize = 1024;
		const batchCount = 4;
		const complexSize = fftSize / 2 + 1;
		const input = createRandomSignal(fftSize * batchCount);

		const re = new Float32Array(complexSize * batchCount);
		const im = new Float32Array(complexSize * batchCount);
		const ret1 = addon.batchFftInto(input, re, im, fftSize, batchCount);
		assert(ret1 === undefined, `batchFftInto return=${ret1}, expected undefined`);

		const output = new Float32Array(fftSize * batchCount);
		const ret2 = addon.batchIfftInto(re, im, output, fftSize, batchCount);
		assert(ret2 === undefined, `batchIfftInto return=${ret2}, expected undefined`);

		assertRoundTrip(input, output, 1e-5);
	});

	test("batchFftInto matches batchFft output", () => {
		const fftSize = 512;
		const batchCount = 3;
		const complexSize = fftSize / 2 + 1;
		const input = createRandomSignal(fftSize * batchCount);

		const { re: reRef, im: imRef } = addon.batchFft(input, fftSize, batchCount);
		const re = new Float32Array(complexSize * batchCount);
		const im = new Float32Array(complexSize * batchCount);
		addon.batchFftInto(input, re, im, fftSize, batchCount);

		const errorRe = maxError(reRef, re);
		const errorIm = maxError(imRef, im);
		assert(errorRe < 1e-5, `re error ${errorRe}`);
		assert(errorIm < 1e-5, `im error ${errorIm}`);
	});

	test("batchIfftInto matches batchIfft output", () => {
		const fftSize = 512;
		const batchCount = 3;
		const input = createRandomSignal(fftSize * batchCount);
		const { re, im } = addon.batchFft(input, fftSize, batchCount);

		const outRef = addon.batchIfft(re, im, fftSize, batchCount);
		const output = new Float32Array(fftSize * batchCount);
		addon.batchIfftInto(re, im, output, fftSize, batchCount);

		const error = maxError(outRef, output);
		assert(error < 1e-5, `error ${error}`);
	});

	test("batchFftInto throws RangeError on too-small output", () => {
		const fftSize = 256;
		const batchCount = 2;
		const complexSize = fftSize / 2 + 1;
		const input = new Float32Array(fftSize * batchCount);
		const re = new Float32Array(complexSize * batchCount - 1); // one element short
		const im = new Float32Array(complexSize * batchCount);
		let threw = false;
		try {
			addon.batchFftInto(input, re, im, fftSize, batchCount);
		} catch (e) {
			threw = true;
			assert(e instanceof RangeError, `expected RangeError, got ${e.constructor.name}`);
		}
		assert(threw, "expected RangeError to be thrown");
	});

	test("batchIfftInto throws RangeError on too-small output", () => {
		const fftSize = 256;
		const batchCount = 2;
		const complexSize = fftSize / 2 + 1;
		const re = new Float32Array(complexSize * batchCount);
		const im = new Float32Array(complexSize * batchCount);
		const outTime = new Float32Array(fftSize * batchCount - 1); // one element short
		let threw = false;
		try {
			addon.batchIfftInto(re, im, outTime, fftSize, batchCount);
		} catch (e) {
			threw = true;
			assert(e instanceof RangeError, `expected RangeError, got ${e.constructor.name}`);
		}
		assert(threw, "expected RangeError to be thrown");
	});

	// ── 2D tests ────────────────────────────────────────────────────────────

	test("batchFft2D impulse at (0,0) gives flat magnitude spectrum", () => {
		const rows = 4;
		const cols = 4;
		const input = new Float32Array(rows * cols).fill(0);
		input[0] = 1; // impulse at (0,0)
		const { re, im } = addon.batchFft2D(input, rows, cols, 1);
		const complexCols = cols / 2 + 1;
		assert(re.length === rows * complexCols, `re length ${re.length}, expected ${rows * complexCols}`);
		assert(im.length === rows * complexCols, `im length ${im.length}`);
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < complexCols; c++) {
				const magSq = re[r * complexCols + c] ** 2 + im[r * complexCols + c] ** 2;
				assert(Math.abs(magSq - 1) < 1e-4, `magnitude^2 at (${r},${c}) = ${magSq}, expected 1`);
			}
		}
	});

	test("batchFft2D round-trip random 4x4 x 2", () => {
		const rows = 4;
		const cols = 4;
		const batchCount = 2;
		const input = createRandomSignal(rows * cols * batchCount);
		const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
		const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
		assertRoundTrip(input, output, 1e-4);
	});

	test("batchFft2D non-square DFTT shape (rows=16, cols=32, batch=100)", () => {
		// This is the shape engineering-side DFTT will use. Wrong axis order fails here.
		const rows = 16;
		const cols = 32;
		const batchCount = 100;
		const complexCols = cols / 2 + 1;
		const input = createRandomSignal(rows * cols * batchCount);
		const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
		// Verify shape: rows × complexCols per block (R2C reduces innermost cols)
		assert(
			re.length === rows * complexCols * batchCount,
			`re length ${re.length}, expected ${rows * complexCols * batchCount}`,
		);
		assert(im.length === rows * complexCols * batchCount, `im length ${im.length}`);
		const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
		assert(
			output.length === rows * cols * batchCount,
			`output length ${output.length}, expected ${rows * cols * batchCount}`,
		);
		assertRoundTrip(input, output, 1e-3); // fp32 VkFFT over 100 batches may loosen slightly
	});

	test("batchFft2D DC-constant blocks across multiple shapes", () => {
		const shapes = [
			{ rows: 8, cols: 8 },
			{ rows: 16, cols: 32 },
			{ rows: 32, cols: 16 },
			{ rows: 4, cols: 8 },
		];
		const batchCount = 3;
		for (const { rows, cols } of shapes) {
			const complexCols = cols / 2 + 1;
			const v = 0.5;
			const input = new Float32Array(rows * cols * batchCount).fill(v);
			const { re, im } = addon.batchFft2D(input, rows, cols, batchCount);
			// DC bin at (0,0) of each block = v * rows * cols (unnormalized forward DFT)
			const expectedDc = v * rows * cols;
			for (let b = 0; b < batchCount; b++) {
				const dcRe = re[b * rows * complexCols + 0];
				const dcIm = im[b * rows * complexCols + 0];
				assert(
					Math.abs(dcRe - expectedDc) < 1e-2,
					`shape ${rows}x${cols} batch ${b} DC re=${dcRe}, expected ${expectedDc}`,
				);
				assert(Math.abs(dcIm) < 1e-2, `shape ${rows}x${cols} batch ${b} DC im=${dcIm}, expected 0`);
			}
			const output = addon.batchIfft2D(re, im, rows, cols, batchCount);
			const error = maxError(input, output);
			assert(error < 1e-3, `shape ${rows}x${cols} round-trip error ${error}`);
		}
	});
} else {
	console.log("No Vulkan device — skipping all GPU tests (CI expected)");
}

process.exit(failed);
