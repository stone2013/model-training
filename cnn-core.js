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
  return { classes: [...parsed.classes], M: model, epoch: Number(parsed.epoch) || 0 };
}
