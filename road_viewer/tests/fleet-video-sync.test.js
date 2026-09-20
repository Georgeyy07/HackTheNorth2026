import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveFleetVideo} from '../static/fleet-video-sync.js';
const m={variants:{cascade:{cars:[{carID:'a',launch_timestamp:'2026-09-20T12:00:20Z',duration_s:5,source_start:{source_us:10000000}}]}},sessions:[
  {session:'s2',origin_unix_us:10000000,duration_s:2,video_offset_s:0.1,video_duration_s:1,fps:2,frame_pts_s:[0,0.5],url:'/s2'},
  {session:'s3',origin_unix_us:13000000,duration_s:3,video_offset_s:0,video_duration_s:3,fps:1,frame_pts_s:[0,1,2],url:'/s3'}]};
const base=Date.parse('2026-09-20T12:00:20Z');
const at=s=>resolveFleetVideo(m,'cascade','a',base+s*1000);
test('launch delay, uncovered head and tail, recording gap, end',()=>{
  assert.equal(at(-1).state,'waiting'); assert.equal(at(0).reason,'video-unavailable');
  assert.equal(at(1.5).state,'gap'); assert.equal(at(2.5).reason,'recording-gap');
  assert.equal(at(6).state,'finished');
});
test('session switching and backwards scrubbing follow source clock',()=>{
  assert.equal(at(3.5).session,'s3'); assert.equal(at(3.5).currentTime,0.5);
  assert.equal(at(0.85).session,'s2'); assert.equal(at(0.85).frameIndex,1);
  assert.equal(at(0.85).currentTime,0.75);
});
