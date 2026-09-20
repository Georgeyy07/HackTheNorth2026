// Uses the running viewer's recordings; no database writes.
import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
const browser = await chromium.launch({headless:true,args:['--no-sandbox'],
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/potholes', route => route.fulfill({json:[]}));
  await page.route('**/demo_view/app.js', async route => {
    const body = await (await route.fetch()).text();
    await route.fulfill({contentType:'application/javascript',body:body + `
      window.syncTest = {state, openTelemetryInspectorForCar, closeTelemetryInspector,
        togglePlayPause, setPlaybackSpeed, resolveFleetVideo};`});
  });
  await page.goto((process.env.VIEWER_URL || 'http://localhost:8765') + '/demo_view/index.html');
  await page.waitForFunction(() => window.syncTest?.state.vehicleRoutes.size === 4);
  const aligned = async (tolerance = .25) => page.waitForFunction(tolerance => {
    const d = syncTest, s = d.state, v = document.querySelector('#inspector-video');
    const target = d.resolveFleetVideo(s.syncManifest,s.selectedScenario,s.inspector.carID,s.currentTimeMs);
    return v.readyState >= 2 && !v.seeking && v.currentSrc.endsWith(target.url) &&
      Math.abs(v.currentTime - target.currentTime) < tolerance;
  }, tolerance, {timeout:10000});
  // This used to lose the initial seek when the target was <2.5s from zero.
  await page.evaluate(() => {
    const d = syncTest;
    d.state.currentTimeMs = d.state.startTimeMs + 1000;
    d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()].sort()[1]);
  });
  await aligned(.05);
  assert.equal(await page.locator('#inspector-video').evaluate(v => v.paused), true);
  for (const speed of [1, 5]) {
    await page.evaluate(speed => {
      const d = syncTest;
      d.state.currentTimeMs = d.state.startTimeMs + 60000;
      d.setPlaybackSpeed(speed);
      if (!d.state.isPlaying) d.togglePlayPause();
    }, speed);
    await aligned();
    await page.waitForTimeout(1500);
    await aligned();
    await page.waitForFunction(() => {
      const s = syncTest.state;
      return s.inspector.imuSamples.length > 900 &&
        Math.abs(s.currentTimeMs - s.inspector.sampleCenterMs) < 5000;
    });
    // A stall under the old 2.5s threshold must recover promptly.
    await page.locator('#inspector-video').evaluate(v => {v.currentTime -= 1.5;});
    await aligned();
    await page.evaluate(() => syncTest.togglePlayPause());
    await aligned(.05);
    assert.equal(await page.locator('#inspector-video').evaluate(v => v.paused), true);
  }
  // Same-file car changes must seek even though the URL is unchanged.
  await page.evaluate(() => {
    const d = syncTest;
    d.state.selectedScenario = 'cascade';
    d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()].sort()[0]);
  });
  await aligned(.05);
  await page.evaluate(() => syncTest.openTelemetryInspectorForCar([...syncTest.state.vehicleRoutes.keys()].sort()[1]));
  await aligned(.05);
  await page.evaluate(() => syncTest.closeTelemetryInspector());
  assert.equal(await page.locator('#inspector-video').evaluate(v => v.paused), true);
  assert.deepEqual(errors, []);
  console.log('Passed: metadata alignment, 1x/5x playback, IMU refresh, drift recovery, pause, same-file car switch, and close.');
} finally {await browser.close();}
