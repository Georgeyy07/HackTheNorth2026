// Run against an upload-enabled server and provide two local CSV fixtures.
import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';

const base = process.env.VIEWER_URL || 'http://127.0.0.1:8766';
const [withGps, withoutGps] = process.argv.slice(2);
if (!withGps || !withoutGps) throw new Error('Usage: node tests/upload-browser.mjs with-gps.csv without-gps.csv');
await mkdir('test-results', {recursive: true});
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 1080}});
const errors = [], results = [];
page.on('pageerror', error => errors.push(error.message));
await page.route('https://tile.openstreetmap.org/**', route => route.abort());
const snapshot = () => page.evaluate(() => window.replaySnapshot());
const seek = time => page.locator('#seek').evaluate((input, value) => {
  input.value = String(value); input.dispatchEvent(new Event('input', {bubbles: true}));
}, time);
try {
  await page.goto(base+'/?autoplay=0');
  await page.locator('#upload-panel').waitFor({state: 'visible'});
  for (const [fixture, gps] of [[withGps, true], [withoutGps, false]]) {
    const previous = (await snapshot())?.session;
    await page.locator('#upload-file').setInputFiles(fixture);
    await page.locator('#upload-submit').click();
    await page.waitForFunction(old => window.replaySnapshot?.()?.session && window.replaySnapshot().session !== old,
      previous, {timeout: 180000});
    await seek(0);
    let state = await snapshot();
    assert.equal(state.finalCount, 0); assert.equal(state.events, 0);
    await seek(.47); assert.equal((await snapshot()).finalCount, 0);
    await seek(.48); state = await snapshot();
    assert.equal(state.finalCount, 1);
    assert.ok(state.maxVisibleAvailability <= state.time);
    assert.equal(await page.locator('#terrible-legend').isVisible(), false);
    assert.match(await page.locator('#iri').textContent(), /Good|Medium|Bad/);
    assert.match(await page.locator('#quality-unit').textContent(), /class/);
    await seek(state.duration); state = await snapshot();
    const payload = await (await fetch(base+'/api/session/'+state.session)).json();
    assert.equal(state.sampleIndex+1, payload.session.samples);
    assert.equal(state.finalCount, Math.floor(payload.session.samples/16)-2);
    assert.equal(state.provisionalCount, 2);
    assert.equal(state.carVisible, gps);
    assert.equal(await page.locator('#gyro-missing').isVisible(), true);
    assert.ok(state.maxVisibleAvailability <= state.time);
    await page.screenshot({path: `test-results/upload-${gps ? 'gps' : 'no-gps'}.png`, fullPage: true});
    results.push({session: state.session, samples: state.sampleIndex+1, finalPatches: state.finalCount, gps});
  }
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path: 'test-results/upload-mobile.png', fullPage: true});
  assert.deepEqual(errors, []);
  await writeFile('test-results/upload-browser-results.json', JSON.stringify({passed: true, results, errors}, null, 2)+'\n');
  console.log(JSON.stringify({passed: true, results, errors}));
} finally {
  await browser.close();
}
