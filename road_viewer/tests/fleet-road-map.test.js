import test from 'node:test';
import assert from 'node:assert/strict';
import {SharedRoadMap} from '../static/fleet-road-map.js';
const car={source_start:{source_us:1000000},launch_delay_s:0};
const p=(time,lat=43.47,lon=-80.54,imu=false,yolo=null,quality='good')=>[time,lat,lon,imu,yolo,quality];
test('repeated virtual cars contribute one observation and do not inflate detections',()=>{
 const s=new SharedRoadMap(),a=p(0,43.47,-80.54,true,true,'bad');s.add(car,a);
 const b={source_start:{source_us:1000000},launch_delay_s:20};s.add(b,[20,...a.slice(1)]);
 assert.equal(s.seen.size,1);assert.equal(s.summary().imuLocations,1);assert.equal(s.summary().yoloLocations,1);
 assert.deepEqual([...s.nodes.values()][0].quality,[0,0,1]);
});
test('shared road quality uses recorded grades; positive evidence survives later negatives',()=>{
 const s=new SharedRoadMap();s.add(car,p(0,43.47,-80.54,true,null,'good'));s.add(car,p(.16,43.47,-80.54,false,false,'bad'));
 assert.equal(s.grade([...s.nodes.values()]),'bad');s.add(car,p(.32));assert.equal(s.grade([...s.nodes.values()]),'good');
 assert.equal(s.summary().imuLocations,1);assert.equal(s.summary().yoloLocations,0);
});
test('edges are shared and GPS gaps do not acquire connecting road segments',()=>{
 const s=new SharedRoadMap(),a=p(0),b=p(.16,43.47015);s.add(car,a);s.add(car,b,a);assert.equal(s.edges.size,1);
 s.add(car,p(4,43.4703),b);assert.equal(s.edges.size,1);
 s.reset();assert.deepEqual(s.summary(),{cells:0,edges:0,uniqueObservations:0,imuLocations:0,yoloLocations:0,quality:{good:0,medium:0,bad:0}});
});
