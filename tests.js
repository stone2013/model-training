import test from 'node:test';
import assert from 'node:assert/strict';
import { captureLabel, chooseDetection, isBackgroundClass, modelFromJSON, modelToJSON, partitionDataset } from './cnn-core.js';

test('upload callbacks retain the class selected when upload began', () => {
  let selected = 0;
  const attach = captureLabel(selected);
  selected = 1;
  assert.equal(attach({ file: 'late-decode.png' }).y, 0);
});

test('training and validation membership stays stable across repeated epochs', () => {
  const samples = Array.from({ length: 40 }, (_, index) => ({ id: `sample-${index}`, y: index % 2 }));
  const first = partitionDataset(samples, 20);
  for (let i = 0; i < 10; i++) {
    const next = partitionDataset(samples, 20);
    assert.deepEqual(next.train.map(x => x.id), first.train.map(x => x.id));
    assert.deepEqual(next.validation.map(x => x.id), first.validation.map(x => x.id));
  }
  assert.equal(new Set([...first.train, ...first.validation]).size, samples.length);
});

test('model export/import preserves class names, epoch, and parameters', () => {
  const model = { K: 2, inputChannels: 3, conv: [Array(27).fill(.1)], W: [Array(4).fill(.2), Array(4).fill(.3)] };
  const saved = modelToJSON(['土豆', '黄瓜'], model, 7);
  const loaded = modelFromJSON(JSON.stringify(saved));
  assert.deepEqual(loaded.classes, ['土豆', '黄瓜']);
  assert.equal(loaded.epoch, 7);
  assert.deepEqual(loaded.M, model);
});

test('legacy grayscale models are upgraded to RGB weights on load', () => {
  const old = { classes: ['A', 'B'], epoch: 3, M: { K: 2, conv: [Array(9).fill(1)], W: [[], []] } };
  const loaded = modelFromJSON(old);
  assert.equal(loaded.M.conv[0].length, 27);
  assert.deepEqual(loaded.M.conv[0].slice(0, 3), [0.299, 0.587, 0.114]);
});

test('background and uncertain patches are rejected by multi-object detection', () => {
  assert.equal(isBackgroundClass('背景 / 非目标'), true);
  assert.equal(chooseDetection([0.96, 0.02, 0.02], ['背景 / 非目标', '土豆', '黄瓜'], 0.7, 0.15), null);
  assert.equal(chooseDetection([0.48, 0.45, 0.07], ['土豆', '黄瓜', '背景'], 0.4, 0.1), null);
  assert.equal(chooseDetection([0.82, 0.10, 0.08], ['土豆', '黄瓜', '背景'], 0.7, 0.15)?.index, 0);
});
