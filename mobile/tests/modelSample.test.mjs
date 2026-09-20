import test from 'node:test';
import assert from 'node:assert/strict';
import { modelSample } from '../src/motion/modelSample.js';
import { rotate } from '../src/motion/vqf.js';

const matrix = [1,0,0,0,1,0,0,0,1];
const base = { time: 0, availableAt: 1700000000000, segment: 0, quaternion: [1,0,0,0],
  earthAccel: [1,2,9.81], earthGyro: [.1,.2,.3], mask: [true,true,true,true], speedMps: 0 };
test('model samples preserve SI, gravity, speed zero and source time zero', () => {
  const s = modelSample(base,matrix);
  assert.deepEqual([s.accel_x,s.accel_y,s.accel_z,s.speed], [1,2,9.81,0]);
  assert.equal(s.time,0);
});
test('yaw rotation cancels in horizontal vehicle frame', () => {
  const q = [Math.cos(.7),0,0,Math.sin(.7)];
  const s = modelSample({...base,quaternion:q,earthAccel:rotate(q,base.earthAccel)},matrix);
  [s.accel_x,s.accel_y,s.accel_z].forEach((v,i) => assert.ok(Math.abs(v-base.earthAccel[i])<1e-10));
});
test('missing speed stays missing and unobservable forward direction masks acceleration', () => {
  assert.equal(modelSample({...base,mask:[true,true,true,false]},matrix).speed,null);
  const q = [Math.SQRT1_2,0,Math.SQRT1_2,0];
  assert.equal(modelSample({...base,quaternion:q},matrix).accel_x,null);
});
