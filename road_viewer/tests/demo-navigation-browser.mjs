// Run against the viewer: CHROMIUM_PATH=/path/to/chrome node road_viewer/tests/demo-navigation-browser.mjs
import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
const browser=await chromium.launch({headless:true,args:['--no-sandbox'],
  ...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
    window.audioPlays=[];
    HTMLMediaElement.prototype.play=function(){window.audioPlays.push(this.src);return Promise.resolve();};
  });
  await page.route('**/api/simulated-car-observations?*',r=>r.fulfill({json:[
    {carID:'sim-staggered-01',timestamp:1000,latitude:43.47,longitude:-80.54,road_quality:'good'},
    {carID:'sim-staggered-01',timestamp:10000,latitude:43.471,longitude:-80.54,road_quality:'good'},
  ]}));
  await page.route('**/api/potholes',r=>r.fulfill({json:[]}));
  await page.goto((process.env.VIEWER_URL||'http://localhost:8765')+'/demo_view/index.html');
  const select=async mode=>{
    await page.selectOption('#scenario-filter',mode);
    await page.waitForFunction(mode=>window.navigationSnapshot?.()?.mode===mode,mode);
  };
  const seek=async sec=>page.locator('#timeline-slider').evaluate((el,sec)=>{
    el.value=sec*1000;el.dispatchEvent(new Event('input'));
  },sec);
  const inspectPothole = async () => {
    const responsePromise = page.request.get((process.env.VIEWER_URL||'http://localhost:8765') + '/api/session/session3/imu-window?time_s=81.28');
    await page.locator('.navigation-car').filter({hasText:'⚠'}).first().click();
    const response = await responsePromise;
    assert.equal(response.status(),200);
    const source = await response.json();
    assert.equal(source.session,'session3');
    assert.ok(source.samples.length>900);
    assert.ok(source.samples.every(s=>Math.abs(s.rel_s)<=5));
    await page.waitForFunction(()=>document.querySelector('#inspector-video').readyState>=2);
    await page.waitForFunction(center=>{
      const v=document.querySelector('#inspector-video');
      return !v.seeking && Math.abs(v.currentTime-center)<.2;
    },source.center_s-5-source.video_offset_s);
    assert.equal(await page.locator('#telemetry-inspector').isVisible(),true);
    await page.locator('#map').click({position:{x:30,y:350}});
    assert.equal(await page.locator('#telemetry-inspector').isVisible(),false);
    assert.equal(await page.locator('#inspector-video').evaluate(v=>v.paused),true);
  };
  await select('reroute');
  assert.equal(await page.locator('#navigation-panel').isVisible(),true);
  await seek(18);
  assert.equal(await page.evaluate(()=>navigationSnapshot().state.route_id),'recorded');
  assert.deepEqual(await page.evaluate(()=>audioPlays),[]);
  await page.locator('[data-speed="5"]').click();
  await page.locator('#btn-play-pause').click();
  await page.waitForFunction(()=>navigationSnapshot().time>20);
  await page.locator('#btn-play-pause').click();
  assert.equal(await page.evaluate(()=>navigationSnapshot().state.route_id),'bypass');
  assert.equal((await page.evaluate(()=>audioPlays)).filter(s=>s.endsWith('/rerouting.mp3')).length,1);
  assert.equal(await page.locator('#nav-decision').isVisible(),true);
  await inspectPothole();
  await page.screenshot({path:'/tmp/demo-integrated-reroute.png'});
  await page.locator('#btn-reset').click();
  assert.equal(await page.evaluate(()=>navigationSnapshot().state.route_id),'recorded');
  assert.equal(await page.locator('#nav-events').textContent(),'');
  await select('warning');
  await seek(129);
  await page.locator('#btn-play-pause').click();
  await page.waitForFunction(()=>navigationSnapshot().time>132);
  await page.locator('#btn-play-pause').click();
  assert.match(await page.locator('#nav-alert').textContent(),/200/);
  assert.equal((await page.evaluate(()=>audioPlays)).filter(s=>s.endsWith('/pothole_200m.mp3')).length,1);
  await inspectPothole();
  const plays=(await page.evaluate(()=>audioPlays)).length;
  await seek(180);
  assert.equal((await page.evaluate(()=>audioPlays)).length,plays,'seeking is silent');
  await seek(155);
  await page.locator('#nav-voice').uncheck();
  await page.locator('#btn-play-pause').click();
  await page.waitForFunction(()=>navigationSnapshot().time>159);
  await page.locator('#btn-play-pause').click();
  assert.equal((await page.evaluate(()=>audioPlays)).length,plays,'mute prevents playback');
  await page.selectOption('#scenario-filter','staggered');
  await page.waitForFunction(()=>document.querySelector('#stat-waypoints').textContent==='2');
  assert.equal(await page.locator('#navigation-panel').isVisible(),false);
  assert.equal(await page.locator('#vehicle-filter').isEnabled(),true);
  assert.equal(await page.evaluate(()=>navigationSnapshot()),null);
  assert.deepEqual(errors,[]);
  console.log('Passed: integrated reroute, 200m warning, correct MP3 once per event, silent scrub, mute, reset, source-aligned pothole inspectors, map dismissal, and fleet return.');
} finally {await browser.close();}
