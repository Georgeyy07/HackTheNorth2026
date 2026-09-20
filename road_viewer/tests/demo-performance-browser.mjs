// Run against the viewer with VIEWER_URL and optional CHROMIUM_PATH.
// All API calls are intercepted: this test never writes to Tiger Data.
import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
const browser = await chromium.launch({headless:true, args:['--no-sandbox'],
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
  const page = await browser.newPage();
  const errors = [];
  const writes = [];
  page.on('pageerror', e => errors.push(e.message));
  const start = Date.parse('2026-09-20T12:00:00Z');
  const rows = Array.from({length:400}, (_, i) => ({
    carID:`sim-staggered-0${1 + Math.floor(i / 100)}`,
    timestamp:start + i % 100 * 1000,
    latitude:43.472 + i % 100 * .00001, longitude:-80.54,
    road_quality:['good','medium','bad'][i % 3],
    imu_defect_detected:false, yolo_pothole_detected:false,
  }));
  await page.route('**/api/**', route => {
    if (route.request().method() !== 'GET') writes.push(route.request().url());
    const path = new URL(route.request().url()).pathname;
    return route.fulfill({json:path.endsWith('simulated-car-observations') ? rows :
      path.endsWith('fleet-sync') ? {start:'2026-09-20T12:00:00Z'} : []});
  });
  await page.route('**/demo_view/app.js', async route => {
    const body = await (await route.fetch()).text();
    await route.fulfill({contentType:'application/javascript', body:body + `
      window.demoTest = {state, get trailLayer() {return trailLayer;}, updateMapToCurrentTime, invalidateSegmentCaches,
        openTelemetryInspectorForCar, closeTelemetryInspector, syncLiveInspectorVideo, processObservations,
        get potholeLayer() {return potholeLayer;}};`});
  });
  await page.goto((process.env.VIEWER_URL || 'http://localhost:8765') + '/demo_view/index.html');
  await page.waitForFunction(() => window.demoTest?.state.vehicleRoutes.size === 4);
  const geometry = await page.evaluate(() => {
    const d = window.demoTest;
    const seek = ms => {d.state.currentTimeMs = d.state.startTimeMs + ms;
      d.invalidateSegmentCaches(); d.updateMapToCurrentTime();
      return d.trailLayer.getLayers().length;};
    const initial = seek(0), middle = seek(50000), rewind = seek(0);
    d.state.selectedVehicle = '01'; d.updateMapToCurrentTime();
    const filtered = d.trailLayer.getLayers().length;
    d.state.selectedVehicle = 'ALL'; d.updateMapToCurrentTime();
    return {initial,middle,rewind,filtered,restored:d.trailLayer.getLayers().length};
  });
  assert.deepEqual(geometry, {initial:4,middle:204,rewind:4,filtered:1,restored:4});
  const loads = await page.evaluate(() => {
    const d = window.demoTest, video = document.querySelector('#inspector-video');
    // Reproduce resource selection still pending: currentSrc remains empty.
    Object.defineProperty(video, 'currentSrc', {get:() => ''});
    Object.defineProperty(video, 'readyState', {get:() => 0});
    let loads = 0;
    video.load = () => loads++;
    d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()][0]);
    for (let i=0; i<120; i++) d.syncLiveInspectorVideo(false);
    d.closeTelemetryInspector();
    return loads;
  });
  assert.equal(loads, 1, 'pending video load must not restart each frame');
  const discoveries = await page.evaluate(() => {
    const d = window.demoTest, start = d.state.startTimeMs;
    const point = (sec, car, lat, detected) => ({timestamp:start+sec*1000,
      carID:car, latitude:lat, longitude:-80.54, road_quality:'good',
      imu_defect_detected:detected, yolo_pothole_detected:detected});
    d.state.allObservations = [point(0,'01',43.47,false), point(10,'01',43.47,true),
      point(20,'02',43.47,true), point(30,'01',43.48,true), point(40,'01',43.48,false)];
    d.processObservations();
    const snapshot = sec => {
      d.state.currentTimeMs = start + sec*1000;
      d.invalidateSegmentCaches(); d.updateMapToCurrentTime();
      return {markers:d.potholeLayer.getLayers().length, hits:d.state.potholes.map(p=>p.hit_count)};
    };
    const results = [snapshot(0), snapshot(9.999), snapshot(10), snapshot(25),
      snapshot(40), snapshot(15), snapshot(0), snapshot(40)];
    d.processObservations(); // reload/scenario reset must also clear discoveries
    results.push({markers:d.potholeLayer.getLayers().length, hits:d.state.potholes.map(p=>p.hit_count)});
    return results;
  });
  assert.deepEqual(discoveries, [
    {markers:0,hits:[]}, {markers:0,hits:[]}, {markers:1,hits:[1]},
    {markers:1,hits:[2]}, {markers:2,hits:[2,1]}, {markers:1,hits:[1]},
    {markers:0,hits:[]}, {markers:2,hits:[2,1]}, {markers:0,hits:[]},
  ]);
  assert.deepEqual(writes, [], 'replay must not write duplicate detections');
  assert.deepEqual(errors, []);
  console.log('Passed: lazy roads, forward/backward scrubbing, hide/show, one pending video load, timestamped discoveries, merged hits, rewind, reset, and read-only replay.');
} finally {await browser.close();}
