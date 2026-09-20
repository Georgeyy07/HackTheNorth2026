const $=id=>document.getElementById(id);
const map=L.map('map').setView([43.476,-80.534],15);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const qualityLayer=L.layerGroup().addTo(map),endpointLayer=L.layerGroup().addTo(map);
const routesLayer=L.layerGroup().addTo(map),hazardLayer=L.layerGroup().addTo(map),scoutLayer=L.layerGroup().addTo(map);
const pin=(label,kind='')=>L.divIcon({className:'',html:`<div class="nav-pin ${kind}">${label}</div>`,iconSize:[34,34],iconAnchor:[17,17]});
const car=L.marker([43.476,-80.54],{icon:pin('You'),zIndexOffset:1000}).addTo(map);
let data=null,t=0,playing=false,last=performance.now(),spoken=new Set(),activeRoute=null,request=0;
const clock=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
function pause(){playing=false;$('play').textContent='Play with voice';window.speechSynthesis?.cancel();}
function speak(message){
 if(!$('voice').checked)return;
 if(!('speechSynthesis' in window)){$('voice-status').textContent='Voice unavailable in this browser; alerts remain visible.';return;}
 const utterance=new SpeechSynthesisUtterance(message);utterance.lang='en-CA';utterance.rate=1;
 utterance.onerror=()=>{$('voice-status').textContent='Voice could not play; visual alert is shown.';};
 window.speechSynthesis.cancel();window.speechSynthesis.speak(utterance);
}
function stateAt(time){return data.replay.states[Math.min(data.replay.states.length-1,Math.floor(time/.25))];}
function pathPosition(coords,f){
 const distances=[0];for(let i=1;i<coords.length;i++)distances.push(distances.at(-1)+map.distance(coords[i-1],coords[i]));
 const target=f*distances.at(-1);let i=1;while(i<distances.length-1&&distances[i]<target)i++;
 const u=(target-distances[i-1])/(distances[i]-distances[i-1]||1);return coords[i].map((v,j)=>coords[i-1][j]+u*(v-coords[i-1][j]));
}
function render(announce=false){
 if(!data)return;const state=stateAt(t),route=data.routes.find(r=>r.id===state.route_id);
 if(activeRoute!==route.id){routesLayer.clearLayers();
  for(const r of data.routes)L.polyline(r.coords,{color:r.id===route.id?'#2875d1':'#929f9b',weight:r.id===route.id?6:3,opacity:.8,dashArray:r.id===route.id?null:'6 8'}).addTo(routesLayer);
  const bypass=data.routes.find(r=>r.id==='bypass');
  const green=[];for(let m=450;m<=1600;m+=10)green.push(pathPosition(bypass.coords,m/bypass.distance_m));
  L.polyline(green,{color:'#1aaf89',weight:4,dashArray:'5 7'}).bindTooltip('Assumed-good surface · synthetic demo prior').addTo(routesLayer);activeRoute=route.id;
 }
 qualityLayer.clearLayers();
 for(const r of data.routes)for(const q of r.quality_regions||[]){
  if(q.grade==='good'||q.available_s>t)continue;
  const points=[];for(let m=q.start_m;m<=q.end_m;m+=5)points.push(pathPosition(r.coords,Math.min(1,m/r.distance_m)));
  L.polyline(points,{color:q.grade==='bad'?'#d14354':'#dea92b',weight:9,opacity:.8}).bindTooltip(`Recorded ${q.grade} road quality`).addTo(qualityLayer);
 }
 car.setLatLng(state.coords);hazardLayer.clearLayers();
 for(const h of data.hazards.filter(h=>h.available_s<=t))L.circleMarker(h.coords,{radius:8,color:'#bd5924',fillColor:'#f39138',fillOpacity:.9,weight:2}).bindPopup(`Recorded YOLO + IMU overlap<br>Session 3: ${h.source_time_s.toFixed(2)} s<br>Reported by ${h.scout_id} at demo ${clock(h.available_s)}`).addTo(hazardLayer);
 scoutLayer.clearLayers();
 for(const [i,s] of data.scouts.entries()){
  const p=s.points.filter(p=>p[0]<=t).at(-1);
  if(p&&t-p[0]<=1)L.marker(p.slice(1),{icon:pin(i?'B':'A','scout')}).bindTooltip(`${s.label}: recorded session 3`).addTo(scoutLayer);
 }
 const bypass=data.routes.find(r=>r.id==='bypass');
 if(t<120)L.marker(pathPosition(bypass.coords,Math.min(.98,.4+t/220)),{icon:pin('S','survey')}).bindTooltip('Simulated survey car · assumed-good bypass').addTo(scoutLayer);
 $('time').textContent=clock(t);$('seek').value=t;$('eta').textContent=clock(state.metrics.eta_s);$('hazards').textContent=state.metrics.potholes;
 $('route-label').textContent=route.label;
 $('guidance').textContent=state.arrived?'Arrived at MidCampus Drive':route.id==='bypass'?'Taking the smoother bypass':'Following the selected route';
 $('quality').textContent=route.id==='bypass'?'Bypass middle section: assumed good. Shared approach can still contain a recorded hazard.':'Surface evidence updates as scouts report. Unobserved road sections remain unknown.';
 const events=data.replay.events.filter(e=>e.time_s<=t);
 const warning=events.filter(e=>e.type==='warning'&&t-e.time_s<9).at(-1);
 $('alert').hidden=!warning;if(warning)$('alert').textContent=warning.message;
 const reroute=events.find(e=>e.type==='reroute');$('decision').hidden=!reroute;
 if(reroute)$('decision').textContent=`Rerouted at ${clock(reroute.time_s)}: ${reroute.before.potholes} → ${reroute.after.potholes} known hazards ahead; ${Math.abs(reroute.after.eta_delta_s).toFixed(1)} seconds ${reroute.after.eta_delta_s<0?'faster':'extra'}. New discoveries can change the remaining count.`;
 $('events').replaceChildren(...events.slice().reverse().map(e=>{const div=document.createElement('div');div.className='event';const b=document.createElement('b');b.textContent=clock(e.time_s);div.append(b,document.createTextNode(e.message));return div;}));
 for(const e of events)if(!spoken.has(e.id)){spoken.add(e.id);if(announce&&['warning','reroute'].includes(e.type))speak(e.message);}
 if(state.arrived)pause();
}
async function load(){const version=++request;pause();data=null;spoken.clear();t=0;activeRoute=null;$('error').hidden=true;$('play').disabled=true;
 try{const r=await fetch(`/api/navigation-demo?mode=${$('mode').value}`);if(!r.ok)throw Error(`Scenario unavailable (${r.status})`);const value=await r.json();if(version!==request)return;data=value;
  $('seek').max=data.replay.duration_s;$('duration').textContent=clock(data.replay.duration_s);map.fitBounds(data.routes.flatMap(r=>r.coords),{padding:[30,30]});
  endpointLayer.clearLayers();
  for(const [point,label] of [[data.routes[0].coords[0],'Start'],[data.routes[0].coords.at(-1),'Destination']])
   L.circleMarker(point,{radius:6,color:'#173e35',fillColor:'white',fillOpacity:1,weight:2}).bindTooltip(label,{permanent:true,direction:'top'}).addTo(endpointLayer);
  render();$('play').disabled=false;
 }catch(e){$('error').hidden=false;$('error').textContent=e.message;}
}
$('play').onclick=()=>{if(!data)return;if(playing){pause();return;}if(t>=data.replay.duration_s){t=0;spoken.clear();}playing=true;last=performance.now();$('play').textContent='Pause';speak('Navigation started. Listening for road reports.');};
$('restart').onclick=()=>{pause();t=0;spoken.clear();render();};
$('mode').onchange=load;
$('seek').oninput=e=>{pause();t=Number(e.target.value);spoken=new Set(data.replay.events.filter(e=>e.time_s<=t).map(e=>e.id));render();};
$('voice').onchange=()=>{if(!$('voice').checked)window.speechSynthesis?.cancel();};
let lastPaint=0;
function tick(now){const dt=Math.min(.25,(now-last)/1000);last=now;if(playing&&data){t=Math.min(data.replay.duration_s,t+dt*Number($('rate').value));if(now-lastPaint>80){render(true);lastPaint=now;}}requestAnimationFrame(tick);}requestAnimationFrame(tick);
window.navigationSnapshot=()=>data?{mode:data.mode,time:t,playing,state:stateAt(t),events:data.replay.events.filter(e=>e.time_s<=t)}:null;
load();
