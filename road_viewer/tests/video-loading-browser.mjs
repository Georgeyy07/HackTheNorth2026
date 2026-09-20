// Run against a viewer export with previews built by scripts.build_video_previews.
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
      window.loadingTest = {state, openTelemetryInspectorForCar, closeTelemetryInspector,
        setPlaybackSpeed, togglePlayPause, resolveFleetVideo};`});
  });
  await page.goto((process.env.VIEWER_URL || 'http://localhost:8765') + '/demo_view/index.html');
  await page.waitForFunction(() => window.loadingTest?.state.vehicleRoutes.size === 4);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Network.enable');
  await cdp.send('Network.setCacheDisabled', {cacheDisabled:true});
  // Cold-cache 4 Mbps connection, 50 ms latency; exercise real video decoding.
  await cdp.send('Network.emulateNetworkConditions', {
    offline:false, latency:50, downloadThroughput:500000, uploadThroughput:100000,
  });
  const timings = [];
  for (const speed of [1, 5]) {
    for (const car of [3, 0, 1, 2]) {
      const start = Date.now();
      await page.evaluate(({car,speed}) => {
        const d = loadingTest;
        d.closeTelemetryInspector();
        d.state.currentTimeMs = d.state.startTimeMs + 43000;
        d.setPlaybackSpeed(speed);
        if (!d.state.isPlaying) d.togglePlayPause();
        d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()].sort()[car]);
      }, {car,speed});
      await page.waitForFunction(() => {
        const d = loadingTest, s = d.state, v = document.querySelector('#inspector-video');
        const target = d.resolveFleetVideo(s.syncManifest,s.selectedScenario,s.inspector.carID,s.currentTimeMs);
        return v.readyState >= 3 && !v.paused && !v.seeking &&
          v.currentSrc.endsWith(target.url) && Math.abs(v.currentTime-target.currentTime) < .5;
      }, null, {timeout:5000});
      const elapsed = Date.now() - start;
      timings.push({car:car+1,speed,ms:elapsed});
      assert.ok(elapsed < 3000, `Car ${car+1} at ${speed}x took ${elapsed}ms to start`);
      // Playback must actually present frames, not remain in a seek loop.
      const before = await page.locator('#inspector-video').evaluate(v => v.getVideoPlaybackQuality().totalVideoFrames);
      await page.waitForTimeout(750);
      assert.ok(await page.locator('#inspector-video').evaluate((v,before) =>
        v.getVideoPlaybackQuality().totalVideoFrames > before, before));
    }
  }
  assert.deepEqual(errors, []);
  console.log('Cold-cache loading at 4 Mbps:', JSON.stringify(timings));
} finally {await browser.close();}
