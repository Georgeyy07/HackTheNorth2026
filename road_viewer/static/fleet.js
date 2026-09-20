import {SharedRoadMap,ROAD_COLORS} from './fleet-road-map.js';
import {upperBound,clock} from './replay.js';
const $=id=>document.getElementById(id);
const colors=['#247dce','#df7148','#8b61bc','#199776','#cc9b22'];
const map=L.map('map').setView([43.473,-80.537],15);
const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
tiles.on('tileerror',()=>{$('tile-warning').hidden=false;});tiles.on('tileload',()=>{$('tile-warning').hidden=true;});
new ResizeObserver(()=>map.invalidateSize({pan:false})).observe($('map'));
const layers=L.layerGroup().addTo(map);
const roadLayer=L.layerGroup().addTo(map),defectLayer=L.layerGroup().addTo(map);
const roadState=new SharedRoadMap(),roadLines=new Map(),roadDots=new Map();
map.createPane('fleetQuality').style.zIndex=410;map.createPane('fleetDefects').style.zIndex=450;
const renderer=L.canvas({padding:.5,pane:'fleetQuality'});let mappedTime=-1;
function resetRoad(){roadState.reset();roadLayer.clearLayers();defectLayer.clearLayers();roadLines.clear();roadDots.clear();cars.forEach(c=>c.lastIndex=-1);mappedTime=-1;}
function roadPopup(node){return `<strong>Shared road observations</strong><br>Quality: ${title(roadState.grade([node]))}<br>IMU disturbance: ${node.imu?'Detected':node.imuKnown?'Not detected':'Unknown'}<br>YOLO pothole: ${node.yolo?'Detected':node.yoloKnown?'Not detected':'Unknown'}<br>${node.observations} unique recorded patches<br><small>Approximate vehicle GPS location · repeated car copies deduplicated</small>`;}
function paintDefects(){
 defectLayer.clearLayers();
 const groups=new Map();
 for(const node of roadState.nodes.values()){
  if(!node.imu&&!node.yolo)continue;
  const pixel=map.project([node.lat,node.lon]),key=`${Math.floor(pixel.x/48)},${Math.floor(pixel.y/48)}`;
  if(!groups.has(key))groups.set(key,[]);groups.get(key).push(node);
 }
 for(const nodes of groups.values()){
  const lat=nodes.reduce((a,n)=>a+n.lat,0)/nodes.length,lon=nodes.reduce((a,n)=>a+n.lon,0)/nodes.length;
  const anchor=nodes.reduce((a,n)=>Math.hypot(n.lat-lat,n.lon-lon)<Math.hypot(a.lat-lat,a.lon-lon)?n:a);
  const imu=nodes.filter(n=>n.imu).length,yolo=nodes.filter(n=>n.yolo).length;
  const markerIcon=L.divIcon({className:'shared-defect-marker',html:`<div class="defect-cluster ${imu?'has-imu':'only-yolo'}">${nodes.length}${yolo?'<i></i>':''}</div>`,iconSize:[24,24],iconAnchor:[12,12]});
  L.marker([anchor.lat,anchor.lon],{icon:markerIcon,pane:'fleetDefects'}).bindTooltip(`${imu} IMU locations · ${yolo} YOLO locations`).bindPopup(`<strong>Detections in this area</strong><br>IMU disturbance: ${imu} mapped cells<br>YOLO pothole: ${yolo} mapped cells<br><small>Nearby detections grouped for visibility. Zoom in for detail. Counts describe GPS cells, not confirmed separate defects.</small>`).addTo(defectLayer);
 }
}
map.on('zoomend',paintDefects);
function paintRoad(){
 const changed=roadState.dirtyNodes.size>0;

 for(const key of roadState.dirtyEdges){
  const edge=roadState.edges.get(key),nodes=edge.nodes.map(k=>roadState.nodes.get(k));
  const color=ROAD_COLORS[roadState.grade(nodes)]||'#98a29a';
  if(!roadLines.has(key))roadLines.set(key,L.polyline(nodes.map(n=>[n.lat,n.lon]),{color,weight:7,opacity:.9,renderer}).addTo(roadLayer));
  const line=roadLines.get(key);line.setStyle({color});line.bindTooltip(`Predicted road quality: ${title(roadState.grade(nodes))}`);
 }
 for(const key of roadState.dirtyNodes){
  const node=roadState.nodes.get(key),position=[node.lat,node.lon],color=ROAD_COLORS[roadState.grade([node])]||'#98a29a';
  if(!roadDots.has(key))roadDots.set(key,L.circleMarker(position,{radius:3.5,weight:0,fillOpacity:.95,fillColor:color,renderer}).bindPopup(()=>roadPopup(node)).addTo(roadLayer));
  roadDots.get(key).setStyle({fillColor:color});
 }
 if(changed)paintDefects();
 roadState.dirtyNodes.clear();roadState.dirtyEdges.clear();
 const summary=roadState.summary();$('mapped-count').textContent=`${summary.cells} road cells mapped`;$('imu-count').textContent=`${summary.imuLocations} IMU locations`;$('yolo-count').textContent=`${summary.yoloLocations} YOLO locations`;
}

// Review overlay uses all recorded observations, independently of replay progress.
const jointLayer=L.layerGroup();
let jointEnabled=new URLSearchParams(location.search).get('evidence')==='both';
let jointRegions=[];
function buildJointRegions(){
 const cells=new Map(),seen=new Set(),grid=new SharedRoadMap();
 for(const car of data.cars)for(const p of car.points){
  const sourceUs=car.source_start.source_us+Math.round((p[0]-car.launch_delay_s)*1e6);
  if(seen.has(sourceUs))continue;seen.add(sourceUs);
  if(p[3]!==true||p[4]!==true)continue;
  const key=grid.key(p);
  if(!cells.has(key))cells.set(key,{lat:p[1],lon:p[2],patches:0,first:p[0],last:p[0]});
  const cell=cells.get(key);cell.patches++;cell.first=Math.min(cell.first,p[0]);cell.last=Math.max(cell.last,p[0]);
 }
 jointRegions=[...cells.values()];
}
function showJointRegions(fitBounds=false){
 jointLayer.clearLayers();
 $('joint-toggle').setAttribute('aria-pressed',String(jointEnabled));
 $('joint-note').hidden=!jointEnabled;
 if(!jointEnabled){map.removeLayer(jointLayer);return;}
 jointLayer.addTo(map);
 for(const r of jointRegions){
  L.circleMarker([r.lat,r.lon],{radius:9,color:'#006e73',weight:3,fillColor:'#19d8cc',fillOpacity:.7,pane:'fleetDefects'})
   .bindTooltip('YOLO + IMU in the same patch')
   .bindPopup(`<strong>YOLO + IMU overlap</strong><br>${r.patches} unique 160 ms patches with both flags true<br>Approximate GPS: ${r.lat.toFixed(6)}, ${r.lon.toFixed(6)}<br>Example replay time: ${clock(r.first)}<br><small>Vehicle GPS in an approximately 8 m cell; not a confirmed pothole coordinate.</small>`).addTo(jointLayer);
 }
 $('joint-note').textContent=`Full-recording review: ${jointRegions.length} approximately 8 m regions, ${jointRegions.reduce((n,r)=>n+r.patches,0)} unique patches where YOLO pothole and IMU defect flags are both true in the same 160 ms patch. Teal circles include future observations and are independent of the replay clock. Click a circle for details.`;
 if(fitBounds&&jointRegions.length)map.fitBounds(jointRegions.map(r=>[r.lat,r.lon]),{padding:[45,45],maxZoom:17});
}
$('joint-toggle').addEventListener('click',()=>{jointEnabled=!jointEnabled;if(jointEnabled)setPlaying(false);showJointRegions(true);});
let data=null,cars=[],time=0,playing=false,lastTick=performance.now(),lastPaint=0,controller=null;
const title=s=>s? s[0].toUpperCase()+s.slice(1):'Unknown';
function setPlaying(value){playing=Boolean(value&&data);lastTick=performance.now();$('play').textContent=playing?'Pause':'Play';}
function icon(index,ended=false){return L.divIcon({className:'fleet-marker',html:`<div class="car-pin ${ended?'ended':''}" style="--car:${colors[index%colors.length]}">${index+1}</div>`,iconSize:[35,35],iconAnchor:[17,17]});}
function render(){
 if(!data)return;
 if(time<mappedTime)resetRoad();
 let active=0,gaps=0,finished=0;
 for(const car of cars){
  const i=upperBound(car.points,time,p=>p[0])-1;
  for(let j=car.lastIndex+1;j<=i;j++)roadState.add(car,car.points[j],car.points[j-1]);
  car.lastIndex=i;
  const p=car.points[i],ended=time>car.points.at(-1)[0],fresh=p&&time-p[0]<=.5;
  const state=i<0?'Waiting to start':ended?'Finished':fresh?'On route':'GPS gap';
  if(ended)finished++;else if(fresh)active++;else if(i>=0)gaps++;
  if(p&&(fresh||ended)){
   car.marker.setLatLng(p.slice(1,3));car.marker.setIcon(icon(car.index,ended));
   if(!layers.hasLayer(car.marker))car.marker.addTo(layers);
  }else if(layers.hasLayer(car.marker))layers.removeLayer(car.marker);
  car.card.querySelector('.car-state').textContent=state;
  const q=car.card.querySelector('.quality'),imu=car.card.querySelector('.imu'),yolo=car.card.querySelector('.yolo');
  const shown=fresh?p:null;
  q.textContent=shown?title(shown[5]):'—';q.className=`quality ${shown?.[5]||''}`;
  for(const [element,value,name] of [[imu,shown?.[3],'IMU'],[yolo,shown?.[4],'YOLO']]){
   element.textContent=`${name}: ${!shown?'—':value==null?'Unknown':value?'Detected':'None'}`;
   element.classList.toggle('detected',value===true);
  }
  if(p)car.marker.setTooltipContent(`Car ${car.index+1} · ${state} · ${shown?title(shown[5]):'no current score'}`);
  car.state=state;car.visibleRow=shown;
 }
 mappedTime=time;paintRoad();
 $('counts').textContent=`${active} driving · ${gaps} GPS gaps · ${finished} finished · ${cars.length} cars total`;
 $('elapsed').textContent=clock(time,true);$('seek').value=String(time);
 $('clock').textContent=new Date(Date.parse(data.start)+time*1000).toISOString().slice(11,19)+' UTC';
}
function seek(t){time=Math.max(0,Math.min(data.duration_s,t));render();if(time>=data.duration_s)setPlaying(false);}
function fit(){const points=cars.flatMap(c=>c.points.filter((p,i)=>i%20===0).map(p=>p.slice(1,3)));if(points.length)map.fitBounds(points,{padding:[30,30]});}
async function load(){
 controller?.abort();const current=controller=new AbortController();setPlaying(false);data=null;
 $('play').disabled=true;$('seek').disabled=true;$('error').hidden=true;
 try{
  const response=await fetch(`/api/fleet/${$('variant').value}`,{signal:current.signal});
  if(!response.ok)throw Error(`Could not load fleet (${response.status})`);
  const result=await response.json();if(current!==controller)return;data=result;
  layers.clearLayers();resetRoad();$('cars').replaceChildren();
  cars=data.cars.map((c,index)=>{
   const color=colors[index%colors.length],card=document.createElement('div');card.className='car-card';card.style.setProperty('--car',color);
   // Model metadata stays in text nodes; car labels are generated locally.
   card.innerHTML=`<div class="car-title"><span class="car-name">Car ${index+1}</span><span class="car-state"></span></div><div class="car-source"></div><div class="readouts"><span class="quality"></span><span class="imu"></span><span class="yolo"></span></div>`;
   const s=c.source_start;card.querySelector('.car-source').textContent=`${s.session.replace('session','Session ')} · ${s.source_start_s.toFixed(2)}s · starts stopped${c.launch_delay_s?` · +${c.launch_delay_s}s`:''}`;
   $('cars').append(card);
   const marker=L.marker(c.points[0].slice(1,3),{icon:icon(index),zIndexOffset:1000+index}).bindTooltip(`Car ${index+1}`);
   return {...c,index,color,card,marker,lastIndex:-1};
  });
  $('duration').textContent=clock(data.duration_s);$('seek').max=String(data.duration_s);$('seek').disabled=false;$('play').disabled=false;
  $('scenario-note').textContent=data.variant==='staggered'?`${data.cars.length} cars launch together at recorded stationary points, then follow the remaining route. Start speeds are zero; each start has at least two seconds of measured rest remaining.`:`Cars follow identical observations with launch delays of ${data.cars.map(c=>c.launch_delay_s).join(', ')} seconds. Each launches at the same stationary location.`;
  buildJointRegions();fit();seek(0);setPlaying(!jointEnabled);showJointRegions(jointEnabled);
 }catch(e){if(e.name==='AbortError')return;$('error').hidden=false;$('error').textContent=e.message;}
}
$('variant').addEventListener('change',load);$('play').addEventListener('click',()=>{if(time>=data.duration_s)seek(0);setPlaying(!playing);});
$('restart').addEventListener('click',()=>{if(data){seek(0);setPlaying(false);}});
$('seek').addEventListener('input',e=>{setPlaying(false);seek(Number(e.target.value));});$('fit').addEventListener('click',fit);
function tick(now){const dt=Math.min(.25,(now-lastTick)/1000);lastTick=now;if(playing&&data){time=Math.min(data.duration_s,time+dt*Number($('rate').value));if(now-lastPaint>100||time>=data.duration_s){seek(time);lastPaint=now;}}requestAnimationFrame(tick);}
requestAnimationFrame(tick);
window.fleetSnapshot=()=>data?{variant:data.variant,time,playing,duration:data.duration_s,sharedMap:roadState.summary(),jointRegions:jointRegions.length,jointReview:jointEnabled,carTrails:0,cars:cars.map(c=>({carID:c.carID,state:c.state,markerVisible:layers.hasLayer(c.marker),point:c.visibleRow,index:c.lastIndex}))}:null;
load();
