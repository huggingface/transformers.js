import './style.css';
import { env, AutoModel, ones } from '@xenova/transformers';
import Chart from 'chart.js/auto';

// Throw an error if WebGPU is not supported
if (!navigator.gpu) {
  const err = 'WebGPU is not supported by this browser.';
  alert(err)
  throw Error(err);
}

env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
env.backends.onnx.wasm.numThreads = 1;

// Reference the elements that we will need
const ctx = document.getElementById('chart');
const batchSizes = document.getElementById('batch-sizes');
const xscale = document.getElementById('x-scale');
const yscale = document.getElementById('y-scale');
const sequenceLength = document.getElementById('sequence-length');
const modelID = document.getElementById('model-id');
const status = document.getElementById('status');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const tests = document.getElementsByClassName('tests');

// Benchmark settings
const NUM_WARMUP_STEPS = 3;
const MODEL_CACHE = new Map();

// Chart configuration
const initChart = () => {
  const config = {
    type: 'line',
    data: {
      labels: [],
      datasets: [],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Batch size',
          },
          min: 1,
        },
        y: {
          title: {
            display: true,
            text: 'Time (ms)',
          },
        }
      }
    },
  };
  const chart = new Chart(ctx, config);
  return chart;
}
let chart = initChart();
const toggleScale = (axis, enabled) => {
  chart.options.scales[axis].type = enabled ? 'logarithmic' : 'linear';
  chart.update();
}

const getSelectedTests = () => {
  return [...tests].filter(x => x.checked);
}

const updateDatasets = () => {
  chart.data.datasets = getSelectedTests().map(test => {
    const color = test.getAttribute('data-color');
    return {
      label: test.value,
      data: [],
      borderColor: `rgba(${color}, 1)`,
      backgroundColor: `rgba(${color}, 0.5)`,
    }
  })
  chart.update();
}
updateDatasets();
[...tests].forEach(test => test.addEventListener('change', updateDatasets));

xscale.addEventListener('change', () => toggleScale('x', xscale.checked));
yscale.addEventListener('change', () => toggleScale('y', yscale.checked));

const generateDummyInputs = (batch_size, seqLength) => {
  const inputs = ones([batch_size, seqLength]);

  const model_inputs = {
    input_ids: inputs,
    attention_mask: inputs,
  }
  return model_inputs;
}

let adapterInfo;
let gpuHasFp16 = false;
try {
  // Shouldn't fail since the WebGPU model has loaded successfully
  const adapter = await navigator.gpu.requestAdapter();
  adapterInfo = await adapter.requestAdapterInfo();
  gpuHasFp16 = adapter.features.has('shader-f16')
} catch (err) {
  adapterInfo = {};
}
if (!gpuHasFp16) {
  const element = document.querySelector('.tests[data-device="webgpu"][data-dtype="fp16"]');
  element.setAttribute('unsupported', true);
  element.disabled = true;
  element.title = 'This device does not support fp16 on WebGPU';
}

status.textContent = 'Ready';

let interrupted = false;
start.addEventListener('click', async () => {
  const validTests = [...tests].filter(test => !test.getAttribute('unsupported'))
  // Update UI
  start.disabled = true;
  stop.disabled = false;
  batchSizes.disabled = true;
  sequenceLength.disabled = true;
  modelID.disabled = true;
  validTests.forEach(test => test.disabled = true);
  interrupted = false;

  // Get parameters
  const model_id = modelID.value;
  const batch_sizes = batchSizes.value.split(',').map(x => parseInt(x)).filter(x => x);
  const seqLength = parseInt(sequenceLength.value);
  const selectedTests = getSelectedTests().map(x => ({
    label: x.value,
    dtype: x.getAttribute('data-dtype'),
    device: x.getAttribute('data-device'),
  }));

  // Reset
  chart.destroy();
  chart = initChart();
  updateDatasets();

  // NOTE: Models must be loaded sequentially (otherwise it will fail due to multiple calls to initWasm())
  const testsToRun = new Map();
  for (const test of selectedTests) {
    const { label, dtype, device, quantized } = test;

    const key = `${model_id}///${label}`;

    const cached = MODEL_CACHE.get(key);
    if (cached) {
      testsToRun.set(label, cached);
      continue;
    }
    status.textContent = 'Loading model(s)...';

    try {
      const model = await AutoModel.from_pretrained(model_id, {
        quantized,
        device,
        dtype,
      });
      MODEL_CACHE.set(key, model);
      testsToRun.set(label, model);
    } catch (err) {
      status.textContent = err.message;
      alert(err.message)
      throw err;
    }
  }

  status.textContent = 'Warming up...';

  // Warm up: This is important for the WebGPU execution provider, which compiles the shaders on first load
  for (let i = 0; i < NUM_WARMUP_STEPS; ++i) {
    const model_inputs = generateDummyInputs(1, seqLength);
    for (const [label, model] of testsToRun) {
      await model(model_inputs);
    }
  }

  status.textContent = 'Running benchmark...';

  for (const batch_size of batch_sizes) {
    if (interrupted) break;

    const model_inputs = generateDummyInputs(batch_size, seqLength);

    const times = []

    for (const [label, model] of testsToRun) {
      const start = performance.now();
      await model(model_inputs);
      const end = performance.now();
      times.push(end - start);
    }

    chart.data.labels.push(batch_size);
    for (let i = 0; i < times.length; ++i) {
      chart.data.datasets[i].data.push(times[i]);
    }
    chart.update();
  }

  // Calculate max speedup:
  if (chart.data.labels.length === 0) return;

  const testNames = [...testsToRun.keys()];
  const table = generateResultsTable(model_id, testNames, chart.data, seqLength);


  // Calculate slowest and fastest times
  let minMaxTimes = [Infinity, 0];
  let minMaxIndices = [0, 0];
  for (let i = 0; i < chart.data.datasets.length; i++) {
    const lastTime = chart.data.datasets[i].data.at(-1);
    if (lastTime < minMaxTimes[0]) {
      minMaxTimes[0] = lastTime;
      minMaxIndices[0] = i;
    }
    if (lastTime > minMaxTimes[1]) {
      minMaxTimes[1] = lastTime;
      minMaxIndices[1] = i;
    }
  }

  const speedup = minMaxTimes[1] / minMaxTimes[0];
  const roundedSpeedup = speedup.toFixed(2);
  const params = new URLSearchParams({
    title: `⚡ WebGPU Benchmark Results (${roundedSpeedup}x speedup)`,
    description: table.outerHTML,
  });

  const paramsStr = params.toString();
  status.innerHTML = `⚡ Done! ${testNames.at(minMaxIndices[0])} is <strong>${roundedSpeedup}x</strong> faster than ${testNames.at(minMaxIndices[1])}! ⚡<br><a href="https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/new?${paramsStr}" target="_blank">Share results</a>`;
  start.disabled = false;
  stop.disabled = true;
  batchSizes.disabled = false;
  sequenceLength.disabled = false;
  modelID.disabled = false;
  validTests.forEach(test => test.disabled = false);
});

start.disabled = false;

stop.addEventListener('click', () => {
  status.textContent = 'Stopping...';
  interrupted = true;
  stop.disabled = true;
});

function generateResultsTable(model_id, testNames, data, sequence_length) {

  const datasets = data.datasets.map(d => d.data);
  const batch_sizes = data.labels;

  const container = document.createElement('div');

  const table = document.createElement('table');
  const thead = table.createTHead();
  const tbody = table.createTBody();

  // Add header row
  const headerRow = thead.insertRow();
  headerRow.insertCell().textContent = 'Batch Size';
  testNames.forEach(model => {
    headerRow.insertCell().textContent = model;
  });

  // Add data rows
  batch_sizes.forEach((batchSize, rowIndex) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = batchSize;
    datasets.forEach(dataset => {
      row.insertCell().textContent = dataset[rowIndex].toFixed(2);
    });
  });

  container.appendChild(table);

  const createBulletPoint = (text) => {
    const li = document.createElement('li');
    li.textContent = text;
    return li;
  }

  // Add other information
  const info = document.createElement('ul');
  info.appendChild(createBulletPoint(`Model: ${model_id}`));
  info.appendChild(createBulletPoint(`Tests run: ${testNames.join(', ')}`));
  info.appendChild(createBulletPoint(`Sequence length: ${sequence_length}`));
  info.appendChild(createBulletPoint(`Browser: ${navigator.userAgent}`));
  info.appendChild(createBulletPoint(`GPU: vendor=${adapterInfo.vendor}, architecture=${adapterInfo.architecture}, device=${adapterInfo.device}, description=${adapterInfo.description}`));
  container.appendChild(info);

  return container;
}
