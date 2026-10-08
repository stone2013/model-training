// Browser-safe helpers shared by CNN Lab and its regression tests.
export function captureLabel(label) {
  const captured = Number(label);
  return image => ({ ...image, y: captured });
}

function hash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

export function partitionDataset(items, validationPercent, seed = 0x434e4e) {
  if (!Array.isArray(items) || items.length < 2) throw new Error('至少需要两张图片才能划分训练集和验证集');
  const ranked = items.map((item, index) => ({ item, index, rank: hash(`${seed}:${item.id ?? index}`) }))
    .sort((a, b) => a.rank - b.rank || a.index - b.index);
  const size = Math.min(items.length - 1, Math.max(1, Math.floor(items.length * Number(validationPercent) / 100)));
  const validation = new Set(ranked.slice(0, size).map(x => x.index));
  return { train: items.filter((_, i) => !validation.has(i)), validation: items.filter((_, i) => validation.has(i)) };
}

export function isBackgroundClass(name) {
  return /^(背景|非目标|background|unknown)(?:\s*[/：:-].*)?$/i.test(String(name).trim());
}

export function chooseDetection(probabilities, classes, threshold = 0.7, minMargin = 0.15) {
  if (!probabilities?.length || probabilities.length !== classes.length) return null;
  const ranked = probabilities.map((score, index) => ({ score, index })).sort((a, b) => b.score - a.score);
  const top = ranked[0], second = ranked[1]?.score ?? 0;
  if (probabilities.some(p => !Number.isFinite(p) || p < 0 || p > 1) || !Number.isFinite(threshold) || !Number.isFinite(minMargin)) return null;
  if (isBackgroundClass(classes[top.index]) || top.score < threshold || top.score - second < minMargin) return null;
  return { index: top.index, score: top.score, margin: top.score - second };
}

export function modelToJSON(classes, model, epoch) {
  return { formatVersion: 2, inputChannels: 3, classes: [...classes], M: model, epoch: Number(epoch) || 0 };
}

export function modelFromJSON(value) {
  const parsed = typeof value === 'string' ? JSON.parse(value) : value;
  if (!parsed || !Array.isArray(parsed.classes) || !parsed.classes.length || !parsed.M || !Array.isArray(parsed.M.conv) || !Array.isArray(parsed.M.W)) {
    throw new Error('模型文件格式无效');
  }
  const model = structuredClone(parsed.M);
  // Migrate models exported by the original 16×16 grayscale trainer.
  if (model.conv[0]?.length === 9) {
    model.conv = model.conv.map(filter => filter.flatMap(weight => [weight * 0.299, weight * 0.587, weight * 0.114]));
    model.inputChannels = 3;
  }
  if (model.conv.some(filter => filter.length !== 27) || model.W.length !== parsed.classes.length) throw new Error('模型参数与类别数量不匹配');
  const finiteRow = (row, n) => Array.isArray(row) && row.length === n && row.every(Number.isFinite);
  if (parsed.classes.some(name => typeof name !== 'string' || !name.trim()) || new Set(parsed.classes).size !== parsed.classes.length ||
      model.K !== parsed.classes.length || model.conv.length !== 4 || !model.conv.every(row => finiteRow(row, 27)) ||
      !finiteRow(model.cb, 4) || !model.W.every(row => finiteRow(row, 196)) || !finiteRow(model.b, model.K)) throw new Error('模型参数维度或数值无效');
  return { classes: [...parsed.classes], M: model, epoch: Number(parsed.epoch) || 0 };
}

// Identical black letterboxing and RGB normalization for full images and crops.
export function preprocess(context, image, x = 0, y = 0, width = image.naturalWidth || image.width, height = image.naturalHeight || image.height) {
  const size = 16, scale = Math.min(size / width, size / height);
  context.clearRect(0, 0, size, size);
  context.fillStyle = '#000'; context.fillRect(0, 0, size, size);
  context.drawImage(image, x, y, width, height, (size - width * scale) / 2, (size - height * scale) / 2, width * scale, height * scale);
  const pixels = context.getImageData(0, 0, size, size).data, result = [];
  for (let i = 0; i < size * size; i++) result.push(pixels[i * 4] / 255, pixels[i * 4 + 1] / 255, pixels[i * 4 + 2] / 255);
  return result;
}
export function suppressBoxes(candidates, limit = 40) {
  const kept = [];
  for (const box of [...candidates].sort((a,b) => b.score-a.score)) {
    const overlap = kept.some(other => {
      const area = Math.max(0, Math.min(box.x+box.w,other.x+other.w)-Math.max(box.x,other.x)) * Math.max(0,Math.min(box.y+box.h,other.y+other.h)-Math.max(box.y,other.y));
      return box.k === other.k && area / (box.w*box.h+other.w*other.h-area) > .3;
    });
    if (!overlap) kept.push(box);
  }
  return kept.slice(0,limit);
}
export function evaluatePredictions(items, classes, threshold, margin) {
  const matrix = classes.map(() => Array(classes.length + 1).fill(0));
  let correct = 0, rejected = 0, backgroundTotal = 0, falsePositives = 0;
  for (const {y, probabilities} of items) {
    if (!Number.isInteger(y) || y < 0 || y >= classes.length) throw new Error('测试标签无效');
    const choice = chooseDetection(probabilities, classes, threshold, margin);
    const predicted = choice?.index ?? classes.length;
    matrix[y][predicted]++;
    if (!choice) rejected++;
    const background = isBackgroundClass(classes[y]);
    if (background) { backgroundTotal++; if (choice) falsePositives++; }
    if ((background && !choice) || predicted === y) correct++;
  }
  return {total: items.length, correct, rejected, backgroundTotal, falsePositives, matrix};
}

export function* scanWindows(width, height, minimum = 36) {
  if (!Number.isFinite(minimum) || minimum < 16) throw new Error('最小目标必须至少为16');
  for (let size = minimum; size <= Math.min(width,height)*.8; size *= 1.45) {
    const side = Math.round(size), step = Math.max(8,Math.round(side*.28));
    for (let y=0;y+side<=height;y+=step) for (let x=0;x+side<=width;x+=step) yield {x,y,w:side,h:side};
  }
}
