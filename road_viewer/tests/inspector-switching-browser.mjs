import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
try {
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/potholes',r=>r.fulfill({json:[]}));
 await page.route('**/demo_view/app.js',async r=>{
  const body=await(await r.fetch()).text();await r.fulfill({contentType:'application/javascript',body:body+'\nwindow.inspectorTest={state,openTelemetryInspectorForPoint,openTelemetryInspectorForCar,closeTelemetryInspector,togglePlayPause};'});
 });
 await page.goto((process.env.VIEWER_URL||'http://localhost:8765')+'/demo_view/index.html');
 await page.waitForFunction(()=>window.inspectorTest?.state.vehicleRoutes.size===4);
 await page.evaluate(()=>{const d=inspectorTest;d.state.currentTimeMs=d.state.startTimeMs+100000;});
 const start=Date.now();
 for(let i=0;i<8;i++){
  await page.evaluate(i=>{const d=inspectorTest,cars=[...d.state.vehicleRoutes.keys()].sort();
   if(i%2)d.openTelemetryInspectorForCar(cars[i%4]);
   else d.openTelemetryInspectorForPoint(cars[i%4],d.state.startTimeMs+90000,43.47,-80.54,true);
  },i);
  await page.waitForTimeout(35);
 }
 await page.evaluate(()=>{const d=inspectorTest;d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()].sort()[1]);});
 await page.waitForFunction(()=>{const s=inspectorTest.state;const v=document.querySelector('#inspector-video');return s.inspector.imuSamples.length>900 && s.inspector.sampleCenterMs===s.currentTimeMs && v.readyState>=2&&!v.seeking&&v.currentSrc.includes('session3');});
 console.log('Eight rapid switches + final car ready (ms):',Date.now()-start);
 await page.waitForTimeout(1700);
 assert.equal(await page.evaluate(()=>inspectorTest.state.inspector.isIncident),false);
 assert.equal(await page.locator('#inspector-video').evaluate(v=>v.paused),true);
 await page.evaluate(()=>inspectorTest.togglePlayPause());
 const oldCenter=await page.evaluate(()=>inspectorTest.state.inspector.sampleCenterMs);
 await page.waitForFunction(old=>inspectorTest.state.inspector.sampleCenterMs>old,oldCenter);
 await page.evaluate(()=>{inspectorTest.togglePlayPause();inspectorTest.closeTelemetryInspector();});
 assert.equal(await page.locator('#inspector-video').evaluate(v=>v.paused),true);
 // Reproduce a slow response while the 5x replay advances every frame.
 // Refreshes must finish, not perpetually cancel one another.
 await page.route('**/api/imu-samples?*',async route=>{
  const response=await route.fetch();
  await new Promise(resolve=>setTimeout(resolve,750));
  await route.fulfill({response}).catch(()=>{});
 });
 await page.locator('[data-speed="5"]').click();
 await page.evaluate(()=>{const d=inspectorTest;d.state.currentTimeMs=d.state.startTimeMs+60000;
  d.togglePlayPause();d.openTelemetryInspectorForCar([...d.state.vehicleRoutes.keys()].sort()[1]);});
 await page.waitForFunction(()=>inspectorTest.state.inspector.imuSamples.length>900,null,{timeout:2500});
 const firstCenter=await page.evaluate(()=>inspectorTest.state.inspector.sampleCenterMs);
 await page.waitForFunction(center=>inspectorTest.state.inspector.sampleCenterMs>center,firstCenter,{timeout:2500});
 await page.evaluate(()=>{inspectorTest.togglePlayPause();inspectorTest.closeTelemetryInspector();});
 assert.deepEqual(errors,[]);
 console.log('Passed: latest selection owns video and IMU, no stale incident playback, live car samples refresh even with 750ms latency at 5x playback, close pauses.');
}finally{await browser.close();}
