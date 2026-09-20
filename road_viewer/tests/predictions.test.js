import test from 'node:test';
import assert from 'node:assert/strict';
import {qualityValue, qualityText, visionFrameAt} from '../static/predictions.js';

test('ordinal confidence becomes a relative score, never fabricated physical IRI', () => {
  const row={valid:true,quality_probability:[.6,.3,.1],quality_grade:0,iri_m_per_km:null};
  assert.equal(qualityValue(row,true),25);
  assert.equal(qualityValue(row,false),null);
  assert.match(qualityText(row,true),/Good.*25\/100/);
  assert.equal(qualityValue({...row,valid:false},true),null);
});
test('camera frames respect offset, sampling cadence, gaps and clip end', () => {
  const frames=[{video_time_s:0},{video_time_s:1},{video_time_s:3}];
  const vision={frames,video_offset_s:.4,duration_s:4,fps:1};
  assert.equal(visionFrameAt(vision,.3),null);
  assert.equal(visionFrameAt(vision,1.3),frames[0]);
  assert.equal(visionFrameAt(vision,1.5),frames[1]);
  assert.equal(visionFrameAt(vision,2.6),null);
  assert.equal(visionFrameAt(vision,4.5),null);
  assert.equal(visionFrameAt({...vision,video_offset_s:null},1.5),null);
});
