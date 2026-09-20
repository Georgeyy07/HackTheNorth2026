/** Navigation presentation on the fleet map; the host owns the replay clock. */
export function createNavigationScenario(map, data, onInspectHazard) {
  const panel = document.getElementById('navigation-panel');
  const $ = name => document.getElementById(`nav-${name}`);
  const group = L.layerGroup().addTo(map);
  const roads = L.layerGroup().addTo(group);
  const evidence = L.layerGroup().addTo(group);
  const markers = L.layerGroup().addTo(group);
  const pin = (label, color) => L.divIcon({className:'', iconSize:[38,38], iconAnchor:[19,19],
    html:`<div class="navigation-car" style="background:${color}">${label}</div>`});
  const car = L.marker(data.routes[0].coords[0], {icon:pin('You','#2875d1'), zIndexOffset:1000}).addTo(group);
  const scouts = data.scouts.map((s,i) => L.marker(s.points[0].slice(1), {icon:pin(i?'B':'A','#8459b4')}).bindTooltip(s.label));
  const routeDistances = new Map(data.routes.map(r => {
    const lengths = [0];
    for(let i=1;i<r.coords.length;i++) lengths.push(lengths.at(-1)+map.distance(r.coords[i-1],r.coords[i]));
    return [r.id,lengths];
  }));
  function position(route, distance) {
    const distances = routeDistances.get(route.id);
    const target = Math.max(0,Math.min(distances.at(-1),distance / route.distance_m * distances.at(-1)));
    let lo=1, hi=distances.length-1;
    while(lo<hi){const mid=(lo+hi)>>1;if(distances[mid]<target)lo=mid+1;else hi=mid;}
    const f=(target-distances[lo-1])/(distances[lo]-distances[lo-1]||1);
    return route.coords[lo].map((v,j)=>route.coords[lo-1][j]+f*(v-route.coords[lo-1][j]));
  }
  for(const [coords,label] of [[data.routes[0].coords[0],'Start'],[data.routes[0].coords.at(-1),'Destination']]) {
    L.circleMarker(coords,{radius:6,color:'#173e35',fillColor:'white',fillOpacity:1})
      .bindTooltip(label,{permanent:true,direction:'top'}).addTo(group);
  }
  let activeRoute, evidenceKey, eventKey, lastTime=-1, spoken=new Set(), snapshot;
  const clock = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
  const audio = new Audio();
  audio.preload = 'auto';
  let audioQueue = [];
  let audioPlaying = false;
  function stopAudio() {
    audioQueue = [];
    audio.pause();
    audio.currentTime = 0;
    audioPlaying = false;
  }
  function playNext() {
    if (!audioQueue.length || !$('voice').checked) {audioPlaying=false;return;}
    audioPlaying = true;
    audio.src = `/api/navigation-audio/${audioQueue.shift()}`;
    audio.play().catch(() => {
      audioPlaying = false;
      audioQueue = [];
      $('voice-status').textContent = 'Audio could not play; the notification is shown on the map.';
    });
  }
  audio.onended = playNext;
  function notify(type) {
    if (!$('voice').checked) return;
    audioQueue.push(type === 'warning' ? 'pothole_200m.mp3' : 'rerouting.mp3');
    if (!audioPlaying) playNext();
  }
  function render(time, announce=false) {
    const t=Math.max(0,Math.min(time,data.replay.duration_s));
    if (announce && Math.floor(t * 10) === Math.floor(lastTime * 10)) return;
    const state=data.replay.states[Math.min(data.replay.states.length-1,Math.floor(t/.25))];
    const route=data.routes.find(r=>r.id===state.route_id);
    const events=data.replay.events.filter(e=>e.time_s<=t);
    if(t<lastTime){spoken=new Set(events.map(e=>e.id));stopAudio();}
    lastTime=t;
    snapshot={mode:data.mode,time:t,state,events};
    if(activeRoute!==route.id){
      roads.clearLayers();
      for(const r of data.routes) L.polyline(r.coords,{color:r.id===route.id?'#2875d1':'#929f9b',
        weight:r.id===route.id?6:3,opacity:.85,dashArray:r.id===route.id?null:'6 8'}).addTo(roads);
      const bypass=data.routes.find(r=>r.id==='bypass');
      for(const q of bypass.quality_regions.filter(q=>q.grade==='good')) {
        const coords=[];for(let m=q.start_m;m<=q.end_m;m+=10) coords.push(position(bypass,m));
        L.polyline(coords,{color:'#1aaf89',weight:4,dashArray:'5 7'})
          .bindTooltip('Assumed-good bypass · demonstration only').addTo(roads);
      }
      activeRoute=route.id;
    }
    // Only rebuild static evidence when a report becomes available, or on rewind.
    const hazards=data.hazards.filter(h=>h.available_s<=t);
    const quality=data.routes.flatMap(r=>(r.quality_regions||[])
      .filter(q=>q.grade!=='good'&&q.available_s<=t).map(q=>({r,q})));
    const key=JSON.stringify([hazards.map(h=>h.id),quality.map(({r,q})=>[r.id,q.start_m])]);
    if(key!==evidenceKey){
      evidence.clearLayers();
      for(const {r,q} of quality){
        const points=[];for(let m=q.start_m;m<=q.end_m;m+=5)points.push(position(r,m));
        L.polyline(points,{color:q.grade==='bad'?'#c62828':'#e65100',weight:9,opacity:.8})
          .bindTooltip(`Recorded ${q.grade} surface`).addTo(evidence);
      }
      for(const h of hazards)L.marker(h.coords,{icon:pin('⚠','#c62828'), bubblingMouseEvents:false})
        .on('click', e => { L.DomEvent.stopPropagation(e.originalEvent || e); onInspectHazard?.(h); })
        .bindTooltip(`YOLO + IMU report from ${h.scout_id} at ${clock(h.available_s)} · click for video and IMU`).addTo(evidence);
      evidenceKey=key;
    }
    car.setLatLng(state.coords);
    for(const [i,s] of data.scouts.entries()){
      let lo=0,hi=s.points.length;
      while(lo<hi){const mid=(lo+hi)>>1;if(s.points[mid][0]<=t)lo=mid+1;else hi=mid;}
      const p=s.points[lo-1];
      if(p&&t-p[0]<=1){scouts[i].setLatLng(p.slice(1));if(!markers.hasLayer(scouts[i]))markers.addLayer(scouts[i]);}
      else markers.removeLayer(scouts[i]);
    }
    $('guidance').textContent=state.arrived?'Arrived at MidCampus Drive':route.id==='bypass'?'Taking the smoother bypass':'Following Phillip → University';
    $('eta').textContent=`${clock(state.metrics.eta_s)} remaining · ${state.metrics.potholes} known hazards ahead`;
    const warning=events.filter(e=>e.type==='warning'&&t-e.time_s<9).at(-1);
    $('alert').hidden=!warning;if(warning)$('alert').textContent=warning.message;
    const reroute=events.find(e=>e.type==='reroute');
    $('decision').hidden=!reroute;
    if(reroute)$('decision').textContent=`Rerouted: ${reroute.before.potholes} → ${reroute.after.potholes} hazards ahead; ${Math.abs(reroute.after.eta_delta_s).toFixed(1)} seconds ${reroute.after.eta_delta_s<0?'faster':'extra'}.`;
    const nextKey=events.map(e=>e.id).join('|');
    if(nextKey!==eventKey){
      $('events').replaceChildren(...events.slice().reverse().map(e=>{const item=document.createElement('div');item.textContent=`${clock(e.time_s)} · ${e.message}`;return item;}));
      eventKey=nextKey;
    }
    for(const e of events)if(!spoken.has(e.id)){spoken.add(e.id);if(announce&&['warning','reroute'].includes(e.type))notify(e.type);}
    document.getElementById('stat-potholes').textContent=hazards.length;
    document.getElementById('stat-anomalies').textContent=hazards.length;
  }
  panel.hidden=false;
  $('title').textContent=data.mode==='warning'?'200 m pothole warning':'Automatic safer rerouting';
  $('voice-status').textContent='';
  map.fitBounds(data.routes.flatMap(r=>r.coords),{paddingTopLeft:[40,40],paddingBottomRight:[360,40]});
  render(0);
  return {
    duration:data.replay.duration_s, render,
    nextEvent(t){return data.replay.events.find(e=>e.time_s>t+.2&&['warning','reroute'].includes(e.type))?.time_s;},
    stopAudio,
    snapshot:()=>snapshot,
    dispose(){map.removeLayer(group);panel.hidden=true;stopAudio();},
  };
}
