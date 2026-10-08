import { captureLabel, chooseDetection, modelFromJSON, modelToJSON, partitionDataset } from './cnn-core.js';

const $ = selector => document.querySelector(selector);
let classes = ['类别A', '类别B'];
let selectedClass = 0;
let data = [];
let model = null;
let running = false;
let epoch = 0;
let lossHistory = [];
let nextSampleId = 1;
const N = 16, C = 4, S = 14, P = 7, F = P * P * C, CHANNELS = 3;

function render() {
  const box = $('#classes'); box.replaceChildren();
  classes.forEach((name, index) => {
    const button = document.createElement('button');
    button.textContent = name;
    button.className = index === selectedClass ? 'sel' : '';
    button.onclick = () => { selectedClass = index; render(); };
    box.append(button);
  });
  const counts = Array(classes.length).fill(0);
  data.forEach(sample => counts[sample.y]++);
  $('#info').textContent = `当前：${classes[selectedClass]} ｜ ${counts.map((count, index) => `${classes[index]} ${count}张`).join(' · ')}`;
  $('#count').textContent = data.length;
  const thumbs = $('#thumbs'); thumbs.replaceChildren();
  data.slice(-30).forEach(sample => {
    const tile = document.createElement('div'); tile.className = 'thumb';
    const image = document.createElement('img'); image.src = sample.url; image.alt = classes[sample.y];
    const label = document.createElement('span'); label.textContent = classes[sample.y];
    tile.append(image, label); thumbs.append(tile);
  });
}

function addClass(name) {
  const clean = name.trim();
  if (!clean || classes.includes(clean)) return;
  classes.push(clean); selectedClass = classes.length - 1; model = null; render(); drawNet();
}
$('#addc').onclick = () => { addClass($('#cn').value); $('#cn').value = ''; };
$('#addbg').onclick = () => {
  const existing = classes.findIndex(name => /^(背景|非目标|background|unknown)/i.test(name.trim()));
  if (existing >= 0) selectedClass = existing;
  else addClass('背景 / 非目标');
  render();
};
$('#upload').onclick = () => $('#files').click();
$('#files').onchange = event => {
  // Capture the selected class before any asynchronous image decode starts.
  const attachToCapturedClass = captureLabel(selectedClass);
  [...event.target.files].forEach(file => imageToX(file, (x, url) => {
    data.push({ ...attachToCapturedClass({ x, url, id: `sample-${nextSampleId++}` }) });
    render();
  }));
  event.target.value = '';
};

function imageToX(file, callback) {
  const objectURL = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = N;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = '#000'; context.fillRect(0, 0, N, N);
    const scale = Math.min(N / image.width, N / image.height);
    const width = image.width * scale, height = image.height * scale;
    context.drawImage(image, (N - width) / 2, (N - height) / 2, width, height);
    const pixels = context.getImageData(0, 0, N, N).data, x = [];
    for (let i = 0; i < N * N; i++) {
      x.push(pixels[i * 4] / 255, pixels[i * 4 + 1] / 255, pixels[i * 4 + 2] / 255);
    }
    URL.revokeObjectURL(objectURL); callback(x, URL.createObjectURL(file));
  };
  image.onerror = () => { URL.revokeObjectURL(objectURL); $('#info').textContent = '有图片无法读取，请检查文件格式。'; };
  image.src = objectURL;
}

const randomWeight = scale => (Math.random() * 2 - 1) * scale;
function initModel() {
  model = {
    K: classes.length, inputChannels: CHANNELS,
    conv: Array.from({ length: C }, () => Array.from({ length: 27 }, () => randomWeight(0.25))),
    cb: Array(C).fill(0),
    W: Array.from({ length: classes.length }, () => Array.from({ length: F }, () => randomWeight(0.08))),
    b: Array(classes.length).fill(0)
  };
}
function softmax(values) {
  const max = Math.max(...values), exp = values.map(value => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map(value => value / total);
}
function forward(x) {
  const conv = Array.from({ length: C }, () => Array(S * S).fill(0));
  const pool = Array.from({ length: C }, () => Array(P * P).fill(0));
  const argmax = Array.from({ length: C }, () => Array(P * P).fill(0));
  for (let filter = 0; filter < C; filter++) for (let y = 0; y < S; y++) for (let xx = 0; xx < S; xx++) {
    let value = model.cb[filter];
    for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) for (let ch = 0; ch < CHANNELS; ch++) {
      const wi = (ky * 3 + kx) * CHANNELS + ch;
      value += model.conv[filter][wi] * x[((y + ky) * N + xx + kx) * CHANNELS + ch];
    }
    conv[filter][y * S + xx] = Math.max(0, value);
  }
  for (let filter = 0; filter < C; filter++) for (let y = 0; y < P; y++) for (let xx = 0; xx < P; xx++) {
    let best = -1, bestIndex = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const index = (y * 2 + dy) * S + xx * 2 + dx;
      if (conv[filter][index] > best) { best = conv[filter][index]; bestIndex = index; }
    }
    pool[filter][y * P + xx] = best; argmax[filter][y * P + xx] = bestIndex;
  }
  const flat = pool.flat();
  const logits = model.W.map((row, k) => row.reduce((sum, weight, i) => sum + weight * flat[i], model.b[k]));
  return { conv, pool, argmax, flat, probabilities: softmax(logits) };
}

async function trainBatch(samples) {
  const order = samples.map((_, index) => index).sort(() => Math.random() - 0.5);
  const learningRate = Number($('#lr').value), batchSize = Math.max(1, Number($('#bs').value));
  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize);
    const convGrad = model.conv.map(row => row.map(() => 0));
    const biasConvGrad = Array(C).fill(0), denseGrad = model.W.map(row => row.map(() => 0));
    const biasDenseGrad = Array(model.K).fill(0);
    for (const index of batch) {
      const sample = samples[index], out = forward(sample.x), dz = [...out.probabilities];
      dz[sample.y] -= 1;
      const featureGrad = Array(F).fill(0);
      for (let k = 0; k < model.K; k++) {
        biasDenseGrad[k] += dz[k];
        for (let j = 0; j < F; j++) { denseGrad[k][j] += dz[k] * out.flat[j]; featureGrad[j] += model.W[k][j] * dz[k]; }
      }
      for (let filter = 0; filter < C; filter++) for (let p = 0; p < P * P; p++) {
        const index = out.argmax[filter][p], gradient = featureGrad[filter * P * P + p];
        if (out.conv[filter][index] <= 0) continue;
        const y = Math.floor(index / S), x = index % S;
        biasConvGrad[filter] += gradient;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) for (let ch = 0; ch < CHANNELS; ch++) {
          const wi = (ky * 3 + kx) * CHANNELS + ch;
          convGrad[filter][wi] += gradient * sample.x[((y + ky) * N + x + kx) * CHANNELS + ch];
        }
      }
    }
    const rate = learningRate / batch.length;
    for (let f = 0; f < C; f++) { model.cb[f] -= rate * biasConvGrad[f]; for (let j = 0; j < 27; j++) model.conv[f][j] -= rate * convGrad[f][j]; }
    for (let k = 0; k < model.K; k++) {
      model.b[k] -= rate * biasDenseGrad[k];
      for (let j = 0; j < F; j++) model.W[k][j] -= rate * denseGrad[k][j];
    }
    if (start % (batchSize * 8) === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
}
function metrics(samples) {
  let totalLoss = 0, correct = 0;
  for (const sample of samples) {
    const probabilities = forward(sample.x).probabilities;
    totalLoss -= Math.log(Math.max(1e-9, probabilities[sample.y]));
    if (probabilities.indexOf(Math.max(...probabilities)) === sample.y) correct++;
  }
  return [totalLoss / (samples.length || 1), correct / (samples.length || 1)];
}
async function train(epochs) {
  if (running) return;
  if (new Set(data.map(sample => sample.y)).size < 2) return alert('至少两个类别需要有图片');
  if (data.length < 10) return alert('样本太少，建议先添加更多图片');
  if (!model || model.K !== classes.length) initModel();
  const snapshot = [...data];
  let split;
  try { split = partitionDataset(snapshot, $('#vp').value); }
  catch (error) { return alert(error.message); }
  running = true;
  for (let i = 0; i < epochs && running; i++) {
    await trainBatch(split.train); epoch++;
    const trainMetrics = metrics(split.train), validationMetrics = metrics(split.validation);
    lossHistory.push(trainMetrics[0]);
    $('#ep').textContent = epoch; $('#loss').textContent = trainMetrics[0].toFixed(3);
    $('#acc').textContent = `${(trainMetrics[1] * 100).toFixed(1)}%`;
    $('#val').textContent = `${(validationMetrics[1] * 100).toFixed(1)}%`;
    drawChart(); drawNet();
  }
  running = false;
}
$('#train').onclick = () => train(Number($('#epochs').value));
$('#one').onclick = () => train(1);
$('#stop').onclick = () => { running = false; };
$('#reset').onclick = () => {
  initModel(); epoch = 0; lossHistory = []; $('#ep').textContent = '0';
  $('#loss').textContent = $('#acc').textContent = $('#val').textContent = '—'; drawNet(); drawChart();
};

$('#testup').onclick = () => $('#testfile').click();
$('#testfile').onchange = event => {
  const file = event.target.files[0];
  if (file) imageToX(file, (x, url) => { $('#preview').src = url; $('#preview').dataset.ready = 'true'; });
  event.target.value = '';
};
function rgbPatch(image, x, y, width, height) {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = N;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, x, y, width, height, 0, 0, N, N);
  const pixels = context.getImageData(0, 0, N, N).data, result = [];
  for (let i = 0; i < N * N; i++) result.push(pixels[i * 4] / 255, pixels[i * 4 + 1] / 255, pixels[i * 4 + 2] / 255);
  return result;
}
function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return intersection / (a.w * a.h + b.w * b.h - intersection + 1e-9);
}
async function detectAll() {
  if (!model) return alert('先训练模型');
  const image = $('#preview');
  if (image.dataset.ready !== 'true') return alert('先选择组合图片');
  const width = image.naturalWidth, height = image.naturalHeight;
  const threshold = Number($('#thr').value), minMargin = Number($('#margin').value), minBox = Number($('#minbox').value);
  const candidates = [], sizes = [];
  for (let size = minBox; size <= Math.min(width, height) * 0.8; size *= 1.45) sizes.push(Math.round(size));
  for (let scale = 0; scale < sizes.length; scale++) {
    const size = sizes[scale], step = Math.max(8, Math.round(size * 0.28));
    for (let y = 0; y + size <= height; y += step) for (let x = 0; x + size <= width; x += step) {
      const probabilities = forward(rgbPatch(image, x, y, size, size)).probabilities;
      const choice = chooseDetection(probabilities, classes, threshold, minMargin);
      if (choice) candidates.push({ x, y, w: size, h: size, k: choice.index, score: choice.score });
    }
    $('#answer').textContent = `扫描中 ${scale + 1}/${sizes.length}…`;
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const candidate of candidates) if (!kept.some(other => candidate.k === other.k && intersectionOverUnion(candidate, other) > 0.30)) kept.push(candidate);
  kept.splice(40);
  const canvas = $('#detect'), context = canvas.getContext('2d');
  canvas.width = width; canvas.height = height; canvas.style.display = 'block'; canvas.style.height = 'auto';
  context.drawImage(image, 0, 0, width, height); context.lineWidth = Math.max(2, width / 250); context.font = `${Math.max(14, width / 35)}px monospace`;
  const counts = {};
  kept.forEach(box => {
    counts[classes[box.k]] = (counts[classes[box.k]] || 0) + 1;
    context.strokeStyle = '#62e6bd'; context.strokeRect(box.x, box.y, box.w, box.h);
    const text = `${classes[box.k]} ${Math.round(box.score * 100)}%`, labelY = Math.max(18, box.y - 4);
    context.fillStyle = 'rgba(0,0,0,.75)'; context.fillRect(box.x, labelY - 18, context.measureText(text).width + 8, 20);
    context.fillStyle = '#8ff0cf'; context.fillText(text, box.x + 4, labelY - 3);
  });
  $('#answer').textContent = kept.length ? Object.entries(counts).map(([name, count]) => `${name} ×${count}`).join(' ｜ ') : '未找到可信目标';
  $('#probs').textContent = `通过置信度与类别差距筛选 ${candidates.length} 个候选框；NMS 后 ${kept.length} 个。建议为“背景 / 非目标”类别上传不含目标的区域图片，扫描器会排除该类。`;
}
$('#predict').onclick = detectAll;

function drawNet() {
  if (!model) initModel();
  const canvas = $('#net'), context = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  context.clearRect(0, 0, width, height);
  const labels = ['16×16 RGB', 'CONV 3×3 ×4', 'ReLU + MAXPOOL', 'DENSE', `SOFTMAX ${classes.length}`], xs = [70, 245, 420, 600, 780];
  context.fillStyle = '#8ca4b3'; labels.forEach((label, i) => context.fillText(label, xs[i] - 45, 22));
  for (let layer = 0; layer < 5; layer++) {
    const count = layer === 0 ? 8 : layer === 1 ? 4 : layer === 2 ? 8 : layer === 3 ? 10 : Math.min(classes.length, 9);
    for (let i = 0; i < count; i++) {
      const y = 45 + i * (230 / (count - 1 || 1)); context.beginPath(); context.arc(xs[layer], y, 7, 0, 7); context.fillStyle = '#2c786a'; context.fill();
      if (layer < 4) {
        const nextCount = layer === 0 ? 4 : layer === 1 ? 8 : layer === 2 ? 10 : Math.min(classes.length, 9);
        context.strokeStyle = 'rgba(98,230,189,.09)';
        for (let j = 0; j < nextCount; j++) { const yy = 45 + j * (230 / (nextCount - 1 || 1)); context.beginPath(); context.moveTo(xs[layer] + 8, y); context.lineTo(xs[layer + 1] - 8, yy); context.stroke(); }
      }
    }
  }
}
function drawChart() {
  const canvas = $('#chart'), context = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  context.clearRect(0, 0, width, height); if (lossHistory.length < 2) return;
  const max = Math.max(...lossHistory, 1); context.strokeStyle = '#62e6bd'; context.beginPath();
  lossHistory.forEach((value, index) => { const x = 10 + index * (width - 20) / (lossHistory.length - 1), y = height - 10 - value / max * (height - 20); index ? context.lineTo(x, y) : context.moveTo(x, y); });
  context.stroke();
}
function downloadJSON(value, name) {
  const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(value)], { type: 'application/json' }));
  link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 0);
}
$('#em').onclick = () => model ? downloadJSON(modelToJSON(classes, model, epoch), 'cnn-model.json') : alert('没有模型');
$('#im').onclick = () => $('#mfile').click();
$('#mfile').onchange = async event => {
  try {
    const imported = modelFromJSON(await event.target.files[0].text());
    classes = imported.classes; model = imported.M; epoch = imported.epoch; selectedClass = 0; render(); drawNet();
    $('#ep').textContent = epoch;
  } catch (error) { alert(`导入失败：${error.message}`); }
  event.target.value = '';
};
$('#clearall').onclick = () => { if (confirm('确定清空全部训练图片？')) { data = []; render(); } };
render(); initModel(); drawNet(); drawChart();
