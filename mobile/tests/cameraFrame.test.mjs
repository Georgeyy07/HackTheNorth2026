import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraFrame, frameId } from '../src/motion/cameraFrame.js';
const input = { image:'base64',id:frameId(),deviceId:'device',number:1,timestamp:1700000000000,
  position:{coords:{latitude:43,longitude:-80},timestamp:1700000000000,receivedAt:1700000000000} };
test('fresh capture carries retry identity and causal GPS',()=>{
  const frame=cameraFrame(input);
  assert.match(frame.frame_id,/^[a-f0-9-]{36}$/);
  assert.equal(frame.latitude,43);
  assert.equal(frame.timestamp,input.timestamp);
  assert.deepEqual(cameraFrame(input),frame);
});
test('no cached null image or stale/future position is sent',()=>{
  assert.throws(()=>cameraFrame({...input,image:null}));
  for(const delta of [-4000,1000]) {
    assert.equal(cameraFrame({...input,position:{...input.position,timestamp:input.timestamp+delta}}).latitude,null);
  }
  assert.equal(cameraFrame({...input,position:{...input.position,receivedAt:input.timestamp+1}}).latitude,null);
});
