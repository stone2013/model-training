import test from 'node:test';
import assert from 'node:assert/strict';
import { scanWindows, preprocess, suppressBoxes, evaluatePredictions, captureLabel, chooseDetection, isBackgroundClass, modelFromJSON, modelToJSON, partitionDataset } from './cnn-core.js';

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
  const model = { K: 2, inputChannels: 3, conv: Array.from({length:4},()=>Array(27).fill(.1)), cb:Array(4).fill(0), b:[0,0], W: [Array(196).fill(.2), Array(196).fill(.3)] };
  const saved = modelToJSON(['土豆', '黄瓜'], model, 7);
  const loaded = modelFromJSON(JSON.stringify(saved));
  assert.deepEqual(loaded.classes, ['土豆', '黄瓜']);
  assert.equal(loaded.epoch, 7);
  assert.deepEqual(loaded.M, model);
});

test('legacy grayscale models are upgraded to RGB weights on load', () => {
  const old = { classes: ['A', 'B'], epoch: 3, M: { K: 2, conv: Array.from({length:4},()=>Array(9).fill(1)), cb:Array(4).fill(0), b:[0,0], W: [Array(196).fill(0), Array(196).fill(0)] } };
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

test('shared preprocessing letterboxes rectangular images and uses identical crop transforms', () => {
  const calls = []; const ctx = {clearRect(){},fillRect(){},drawImage(...args){calls.push(args)},getImageData(){return {data:new Uint8ClampedArray(1024).fill(255)}}};
  const image = {width:80,height:40};
  const full = preprocess(ctx,image), crop = preprocess(ctx,image,0,0,80,40);
  assert.deepEqual(calls[0],calls[1]); assert.deepEqual(calls[0].slice(-4),[0,4,16,8]);
  assert.deepEqual(full,crop); assert.equal(full.length,768); assert.equal(full[0],1);
});
test('class mapping and reject-aware matrix distinguish background false positives and target misses', () => {
 const names=['黄瓜','背景','土豆'];
 const result=evaluatePredictions([{y:1,probabilities:[.9,.05,.05]},{y:1,probabilities:[.01,.98,.01]},{y:2,probabilities:[.05,.05,.9]},{y:0,probabilities:[.4,.3,.3]}],names,.7,.15);
 assert.equal(result.correct,2); assert.equal(result.falsePositives,1); assert.equal(result.backgroundTotal,2);
 assert.deepEqual(result.matrix,[[0,0,0,1],[1,0,0,1],[0,0,1,0]]);
});
test('combination detector drops background before per-class NMS', () => {
 const names=['土豆','背景','黄瓜']; const boxes=[];
 for(const [x,p] of [[0,[.99,.005,.005]],[1,[.98,.01,.01]],[50,[.01,.98,.01]],[100,[.01,.01,.98]]]) {
   const choice=chooseDetection(p,names); if(choice) boxes.push({x,y:0,w:20,h:20,k:choice.index,score:choice.score});
 }
 assert.deepEqual(suppressBoxes(boxes).map(b=>b.k),[0,2]);
});
test('malformed models and nonfinite probabilities are rejected', () => {
 assert.throws(()=>modelFromJSON({classes:['A'],M:{K:1,conv:[Array(27).fill(0)],W:[[]]}}));
 assert.equal(chooseDetection([NaN,.9],['A','B']),null);
});

test('actual multiscale window generator bounds crops and rejects invalid settings', () => {
 const windows=[...scanWindows(120,80,36)]; assert.ok(windows.length>0);
 assert.ok(windows.every(b=>b.x>=0 && b.y>=0 && b.x+b.w<=120 && b.y+b.h<=80));
 assert.throws(()=>[...scanWindows(120,80,0)]);
 assert.deepEqual([...scanWindows(10,10,36)],[]);
});
