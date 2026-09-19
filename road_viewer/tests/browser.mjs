import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';

const BASE = process.env.VIEWER_URL || 'http://127.0.0.1:8765';
await mkdir('test-results', {recursive: true});
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 1080}, deviceScaleFactor: 1});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const snapshot = () => page.evaluate(() => window.replaySnapshot());
const seek = async time => page.locator('#seek').evaluate((input, value) => {
  input.value = String(value); input.dispatchEvent(new Event('input', {bubbles: true}));
}, time);
const catalog = await (await fetch(BASE + '/api/catalog')).json();

try {
  await page.goto(BASE + '/?autoplay=0', {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.replaySnapshot?.(), {timeout: 30000});
  let state = await snapshot();
  assert.equal(state.events, 0); assert.equal(state.coloredFixes, 0); assert.equal(state.carVisible, false);
  await seek(.48); state = await snapshot(); assert.equal(state.finalCount, 0);
  await seek(.49); state = await snapshot(); assert.equal(state.finalCount, 1);
  assert.ok(state.maxVisibleAvailability <= state.time);
  await page.locator('#next-event').click(); state = await snapshot();
  assert.equal(state.events, 1); assert.equal(state.mapEvents, 1); assert.equal(state.latestFinal.disturbance, true);
  await page.locator('#play').click();
  const before = (await snapshot()).time;
  await page.waitForFunction(t => window.replaySnapshot().time >= t + .9, before);
  await page.locator('#play').click(); state = await snapshot();
  assert.equal(state.playing, false); assert.ok(state.time > before);
  const paused = state.time;
  await page.waitForTimeout(150); assert.equal((await snapshot()).time, paused);
  await page.locator('#rate').selectOption('25'); await page.locator('#play').click();
  await page.waitForFunction(t => window.replaySnapshot().time >= t + 5, paused);
  await page.locator('#play').click();
  await seek(30); await page.waitForTimeout(300);
  await page.screenshot({path: 'test-results/desktop.png', fullPage: true});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.locator('#about').click(); assert.equal(await page.locator('#about-dialog').evaluate(x => x.open), true);
  await page.locator('#close-about').click();
  await seek(0); state = await snapshot();
  assert.equal(state.events, 0); assert.equal(state.mapEvents, 0); assert.equal(state.coloredFixes, 0);
  assert.equal(await page.locator('#iri').textContent(), '—');

  const results = [];
  for (const recording of catalog.sessions) {
    await page.locator('#session').selectOption(recording.session_id);
    await page.waitForFunction(id => window.replaySnapshot?.()?.session === id, recording.session_id, {timeout: 30000});
    // Selecting a drive autoplays. Seeking pauses and deterministically clears it.
    await seek(0); assert.equal((await snapshot()).events, 0);
    await seek(recording.duration_s); state = await snapshot();
    assert.equal(state.sampleIndex + 1, recording.samples);
    assert.equal(state.finalCount, Math.max(0, Math.floor(recording.samples / 16) - 2));
    assert.equal(state.provisionalCount, recording.samples % 16 ? 3 : 2);
    assert.ok(state.maxVisibleAvailability <= state.time);
    assert.equal(state.playing, false);
    if (recording.dataset === 'lira_cd') assert.equal(await page.locator('#gyro-missing').isVisible(), true);
    results.push({session: state.session, samples: state.sampleIndex + 1, finalPatches: state.finalCount,
                  provisionalPatches: state.provisionalCount, events: state.events, endGpsValid: state.gps.valid});
    if (recording.session_id === 'lira_m13_gm_7245_pass_1') {
      assert.equal(state.gps.valid, false); assert.equal(state.carVisible, false);
      assert.equal(await page.locator('#gps-state').textContent(), 'GPS gap');
      await page.screenshot({path: 'test-results/lira-gps-gap.png', fullPage: true});
    }
  }

  // A narrow-screen user gets the same data and usable controls, without overflow.
  await page.setViewportSize({width: 390, height: 844});
  await page.goto(BASE + '/?autoplay=0&t=30', {waitUntil: 'networkidle'});
  await page.waitForFunction(() => window.replaySnapshot?.()?.time === 30);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path: 'test-results/mobile.png', fullPage: true});
  await page.locator('#next-event').click(); assert.ok((await snapshot()).time > 30);

  // Network loss of street tiles must not stop the dataset or overlay replay.
  const offline = await browser.newPage({viewport: {width: 1280, height: 900}});
  offline.on('pageerror', error => errors.push(error.message));
  await offline.route('https://tile.openstreetmap.org/**', route => route.abort());
  await offline.goto(BASE + '/?autoplay=0&t=5', {waitUntil: 'networkidle'});
  await offline.waitForFunction(() => window.replaySnapshot?.()?.time === 5);
  assert.equal(await offline.locator('#tile-warning').isVisible(), true);
  assert.equal(await offline.evaluate(() => window.replaySnapshot().carVisible), true);
  assert.deepEqual(errors, []);
  await writeFile('test-results/browser-results.json', JSON.stringify({passed: true, browser: await browser.version(),
    sessions: results, samples: results.reduce((n, r) => n + r.samples, 0), errors,
    checked: ['availability gating', 'provisional revisions', 'event timing', 'play/pause', 'speed control',
              'rewind removes future data', 'all nine drives to EOF', 'GPS gaps', 'LiRA missing gyro',
              'desktop/mobile layout', 'map tile failure fallback', 'no browser JS errors']}, null, 2) + '\n');
  console.log(JSON.stringify({passed: true, sessions: results.length, samples: results.reduce((n, r) => n + r.samples, 0), errors}));
} finally {
  await browser.close();
}
