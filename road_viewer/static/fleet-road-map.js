// Shared spatial summary of arrived observations. Replayed copies of the same
// source patch are counted once, even when several virtual cars observe them.
const METRES_PER_DEGREE=111320;
const X_SCALE=METRES_PER_DEGREE*Math.cos(43.47*Math.PI/180);
const NAMES=['good','medium','bad'];
export const ROAD_COLORS={good:'#159c78',medium:'#d2a52e',bad:'#ce4961'};
export class SharedRoadMap {
 constructor(cellMetres=8){this.cellMetres=cellMetres;this.reset();}
 reset(){this.nodes=new Map();this.edges=new Map();this.seen=new Set();this.dirtyNodes=new Set();this.dirtyEdges=new Set();}
 key(p){return `${Math.round(p[2]*X_SCALE/this.cellMetres)},${Math.round(p[1]*METRES_PER_DEGREE/this.cellMetres)}`;}
 node(p){
  const key=this.key(p);
  if(!this.nodes.has(key)){
   const [x,y]=key.split(',').map(Number);
   this.nodes.set(key,{key,lat:y*this.cellMetres/METRES_PER_DEGREE,lon:x*this.cellMetres/X_SCALE,quality:[0,0,0],imu:false,yolo:false,imuKnown:0,yoloKnown:0,observations:0,edges:new Set()});
  }
  return this.nodes.get(key);
 }
 add(car,p,previous){
  const node=this.node(p);
  if(previous&&p[0]-previous[0]<=.5&&p[0]>=previous[0]){
   const distance=Math.hypot((p[1]-previous[1])*METRES_PER_DEGREE,(p[2]-previous[2])*X_SCALE);
   if(distance<=100){
    const prev=this.node(previous);
    if(prev.key!==node.key){
     const keys=[prev.key,node.key].sort(),key=keys.join('|');
     if(!this.edges.has(key))this.edges.set(key,{key,nodes:keys});
     prev.edges.add(key);node.edges.add(key);this.dirtyEdges.add(key);
    }
   }
  }
  const source=car.source_start.source_us+Math.round((p[0]-car.launch_delay_s)*1e6);
  if(this.seen.has(source))return;
  this.seen.add(source);node.observations++;
  const grade=NAMES.indexOf(p[5]);if(grade>=0)node.quality[grade]++;
  if(p[3]!=null)node.imuKnown++;
  if(p[4]!=null)node.yoloKnown++;
  node.imu ||= p[3]===true;node.yolo ||= p[4]===true;
  this.dirtyNodes.add(node.key);for(const edge of node.edges)this.dirtyEdges.add(edge);
 }
 grade(nodes){
  const counts=[0,0,0];for(const node of nodes)node.quality.forEach((n,i)=>counts[i]+=n);
  if(!counts.some(Boolean))return null;
  // Deterministic ties favor the rougher category.
  const max=Math.max(...counts);return NAMES[counts.lastIndexOf(max)];
 }
 summary(){const nodes=[...this.nodes.values()];return {cells:nodes.length,edges:this.edges.size,uniqueObservations:this.seen.size,imuLocations:nodes.filter(n=>n.imu).length,yoloLocations:nodes.filter(n=>n.yolo).length,quality:Object.fromEntries(NAMES.map(g=>[g,nodes.filter(n=>this.grade([n])===g).length]))};}
}
