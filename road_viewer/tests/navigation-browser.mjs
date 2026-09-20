// Run with CHROMIUM_PATH=/path/to/chrome node road_viewer/tests/navigation-browser.mjs
import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
try{
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{window.testSpoken=[];window.speechSynthesis.speak=u=>window.testSpoken.push(u.text);window.speechSynthesis.cancel=()=>{};});
 await page.goto((process.env.VIEWER_URL||'http://localhost:8770')+'/navigation-demo');
 await page.waitForFunction(()=>window.navigationSnapshot?.());
 const seek=async t=>page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},t);
 await seek(18);assert.equal(await page.evaluate(()=>window.navigationSnapshot().state.route_id),'recorded');
 await page.locator('#play').click();await page.waitForFunction(()=>window.navigationSnapshot().time>20);await page.locator('#play').click();
 assert.equal(await page.evaluate(()=>window.navigationSnapshot().state.route_id),'bypass');
 assert.ok((await page.evaluate(()=>window.testSpoken)).some(s=>s.startsWith('Rerouting')));
 await page.selectOption('#mode','warning');await page.waitForFunction(()=>window.navigationSnapshot?.()?.mode==='warning');
 await seek(129);await page.locator('#play').click();await page.waitForFunction(()=>window.navigationSnapshot().time>132);await page.locator('#play').click();
 assert.ok((await page.evaluate(()=>window.testSpoken)).some(s=>s.includes('200 metres')));
 await seek(0);assert.equal((await page.locator('#events').textContent()).trim(),'');assert.deepEqual(errors,[]);
 console.log('Navigation browser checks passed: causal reroute, voice request, warning distance message, rewind.');
}finally{await browser.close();}
