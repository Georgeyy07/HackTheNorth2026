"""Import pinned LiRA-CD recordings and independently measured 100-m IRI.

Raw accelerometer (~50 Hz, g) and speed (km/h) are held forward onto the
existing 100 Hz SI grid. Missing gyroscopes remain masked. GPS is used only
for target alignment. TRAIN/VAL/TEST are fixed by road, never random windows.
Requires h5py, pandas and scipy only during this one-time preparation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__:
    from road_training.tools.prepare_data import checksum, training_statistics, write_json, write_record
else:
    from prepare_data import checksum, training_statistics, write_json, write_record

ROADS = {"CPH1": "train", "M3": "val", "M13": "test"}
POLICY = dict(output_hz=100,accel_max_age_s=.1,speed_max_age_s=.25,
              minimum_sensor_coverage=.9,gps_max_gap_s=3.,gps_max_distance_m=20.,
              section_length_m=100,minimum_section_samples=200,
              minimum_section_span_m=90.,minimum_section_valid_fraction=.95,
              minimum_mean_speed_mps=5.,minimum_p10_speed_mps=3.)


def clean_time(values):
    """Sort timestamps and retain the last finite observation per timestamp."""
    values=np.asarray(values)
    values=values[np.isfinite(values).all(1)]
    values=values[np.argsort(values[:,0],kind="stable")]
    return values[np.r_[np.diff(values[:,0])>0,True]] if len(values) else values


def hold_observations(query, observations, maximum_age):
    """Only observations at or before each query; never fill from the future."""
    if len(observations)<1:raise ValueError("No finite observations")
    index=np.searchsorted(observations[:,0],query,side="right")-1
    safe=np.maximum(index,0)
    age=query-observations[safe,0]
    valid=(index>=0)&(age>=0)&(age<=maximum_age)
    values=observations[safe,1:].copy()
    values[~valid]=0
    source_time=np.where(valid,observations[safe,0],np.nan)
    return values,valid,source_time


def project_reference(reference_latlon, car_latlon):
    """Local polyline projection on the published 10-m station grid.

    Same geometric projection as the project's previous LiRA alignment audit;
    no IRI values participate in locating the car.
    """
    origin=reference_latlon.mean(0)
    scale=np.array([111195.,111195.*np.cos(np.deg2rad(origin[0]))])
    curve=(reference_latlon-origin)*scale;query=(car_latlon-origin)*scale
    _,nearest=cKDTree(curve).query(query,k=min(6,len(curve)))
    if nearest.ndim==1:nearest=nearest[:,None]
    starts=np.clip(np.concatenate([nearest-1,nearest],axis=1),0,len(curve)-2)
    a=curve[starts];direction=curve[starts+1]-a
    fraction=np.clip(((query[:,None]-a)*direction).sum(2)/np.maximum((direction**2).sum(2),1e-10),0,1)
    distance=np.linalg.norm(a+fraction[:,:,None]*direction-query[:,None],axis=2)
    best=distance.argmin(1);row=np.arange(len(query))
    return (starts[row,best]+fraction[row,best])*10.,distance[row,best]


def align_roughness(query, sensor_valid, speed, mapgps, speed_observations, reference_gps, iri):
    """Assign independent IRI to conservatively admitted section traversals.

    query is accelerometer acquisition time, not receipt-grid time. GPS may
    bracket query for offline target alignment; it never creates model inputs.
    """
    chainage,distance=project_reference(reference_gps,mapgps[:,[2,1]])
    dt=np.diff(mapgps[:,0]);delta=np.diff(chainage)
    mid=(mapgps[:-1,0]+mapgps[1:,0])/2
    speed_at_mid,speed_ok,_=hold_observations(mid,speed_observations,POLICY["speed_max_age_s"])
    expected=speed_at_mid[:,0]/3.6*dt
    good=(dt<=POLICY["gps_max_gap_s"]) & (distance[:-1]<=POLICY["gps_max_distance_m"]) & (distance[1:]<=POLICY["gps_max_distance_m"])
    good &= speed_ok & (delta>=-3.) & (delta<=expected+np.maximum(30.,.5*expected))
    index=np.searchsorted(mapgps[:,0],query,side="right")-1
    safe=np.clip(index,0,len(good)-1)
    spatial=(index>=0)&(index<len(good))&good[safe]&np.isfinite(query)
    station=np.interp(query,mapgps[:,0],chainage)
    valid=spatial&sensor_valid
    overall=np.full(len(query),np.nan,np.float32)
    section_ids=np.full(len(query),-1,np.int32)
    sections=[];rejected=Counter()
    for section in range(len(iri)//10):
        lo,hi=section*100.,(section+1)*100.
        use=(station>=lo)&(station<hi)&valid
        ix=np.flatnonzero(use)
        if len(ix)<POLICY["minimum_section_samples"]:
            rejected["insufficient_samples"]+=1;continue
        if np.ptp(station[use])<POLICY["minimum_section_span_m"]:
            rejected["incomplete_spatial_coverage"]+=1;continue
        if len(ix)<POLICY["minimum_section_valid_fraction"]*(ix[-1]-ix[0]+1):
            rejected["gaps_or_multiple_visits"]+=1;continue
        if speed[use].mean()<POLICY["minimum_mean_speed_mps"] or np.quantile(speed[use],.1)<POLICY["minimum_p10_speed_mps"]:
            rejected["stopped_or_slow"]+=1;continue
        target=iri[section*10:(section+1)*10]
        if not np.isfinite(target).all() or (target<=0).any():
            rejected["invalid_iri_reference"]+=1;continue
        target=float(target.mean())
        overall[use]=target;section_ids[use]=section
        sections.append(dict(section=section,start_m=lo,end_m=hi,iri_m_per_km=target,
                             observed_samples=len(ix),first_sample=int(ix[0]),last_sample=int(ix[-1])))
    audit=dict(sections=len(sections),rejections=dict(rejected),
               valid_aligned_fraction=float(valid.mean()),
               gps_reference_distance_m_quantiles=np.quantile(distance,[.5,.95,1]).tolist(),
               rejected_gps_intervals=int((~good).sum()))
    return overall,section_ids,sections,audit


def verify_input(path, receipts):
    with path.open("rb") as handle:actual=hashlib.file_digest(handle,"md5").hexdigest()
    if actual!=receipts[path.name]["md5"]:raise ValueError(f"Pinned source checksum mismatch: {path}")
    return dict(path=str(path),md5=actual,sha256=checksum(path))


def supported_reference(stations, iri, gps_stations, gps_latlon):
    """Keep the reference prefix covered by GPS; never extrapolate coordinates.

    Incomplete trailing 100-m sections are subsequently ignored by alignment.
    A missing prefix would change section indices, so it is rejected explicitly.
    """
    if not (np.diff(gps_stations)>0).all():
        raise ValueError("Reference chainage is not strictly increasing")
    if gps_stations[0]>stations[0]:
        raise ValueError("Reference GPS is missing the start of the IRI grid")
    covered=stations<=gps_stations[-1]
    if covered.sum()<2:raise ValueError("Insufficient overlapping reference stations")
    reference=np.column_stack([np.interp(stations[covered],gps_stations,gps_latlon[:,j]) for j in range(2)])
    if not np.isfinite(reference).all():raise ValueError("Invalid reference coordinates")
    return iri[covered],reference


def load_reference(source, road, receipts):
    csv_path=source/"raw/23097002"/f"{road.lower()}_iri_mpd_rut_hh.csv"
    gps_path=source/"raw/23097002"/f"{road.lower()}_zp_hh.csv"
    hashes=[verify_input(path,receipts) for path in (csv_path,gps_path)]
    table=pd.read_csv(csv_path,sep=";",encoding="cp1252")
    table.columns=table.columns.str.strip()
    stations=table["Distance [m]"].to_numpy()
    iri=table[["IRI (5) [m/km]","IRI (21) [m/km]"]].to_numpy()
    np.testing.assert_allclose(stations,np.arange(len(stations))*10.,atol=1e-6)
    gps=pd.read_csv(gps_path,sep=";",encoding="cp1252",usecols=["Distance [m]","Lat","Lon"])
    original_rows=len(iri)
    iri,reference_gps=supported_reference(stations,iri,gps["Distance [m]"].to_numpy(),gps[["Lat","Lon"]].to_numpy())
    hashes[0].update(original_iri_rows=original_rows,gps_supported_iri_rows=len(iri),
                     excluded_trailing_iri_rows=original_rows-len(iri))
    return iri,reference_gps,hashes


def prepare_pass(staging, road, split, name, group, iri, reference_gps, provenance):
    speed_key="obd.spd_veh" if "obd.spd_veh" in group else "obd.spd"
    missing=[key for key in ("acc.xyz","gps_mapmatch",speed_key) if key not in group]
    audit=dict(road=road,split=split,pass_name=name)
    if missing:return None,dict(audit,status="missing_sensors",missing=missing)
    acc=clean_time(group["acc.xyz"][...])
    spd=clean_time(group[speed_key][...])
    gps=clean_time(group["gps_mapmatch"][...])
    if min(len(acc),len(spd),len(gps))<2:return None,dict(audit,status="insufficient_observations")
    # Remove the large UNIX origin before interpolation/searches; the output
    # grid is exact to floating-point precision in recording-relative seconds.
    origin=float(acc[0,0])
    for values in (acc,spd,gps):values[:,0]-=origin
    t=np.arange(int(np.floor(acc[-1,0]*100))+1,dtype=np.float64)/100.
    av,acc_valid,acc_time=hold_observations(t,acc,POLICY["accel_max_age_s"])
    sv,speed_valid,speed_time=hold_observations(t,spd,POLICY["speed_max_age_s"])
    sensor_valid=acc_valid&speed_valid
    audit["sensor_valid_fraction"]=float(sensor_valid.mean())
    if sensor_valid.mean()<POLICY["minimum_sensor_coverage"]:
        return None,dict(audit,status="low_sensor_coverage")
    x=np.zeros((len(t),7),np.float32);mask=np.zeros_like(x,bool)
    x[:,:3]=av*9.80665;mask[:,:3]=acc_valid[:,None]
    x[:,6]=sv[:,0]/3.6;mask[:,6]=speed_valid
    # No gyroscope or CAN yaw substitute is invented; the encoder masks it.
    x[~mask]=0
    overall,ids,sections,alignment=align_roughness(acc_time,sensor_valid,x[:,6],gps,spd,reference_gps,iri)
    sensor_time=np.column_stack([acc_time,speed_time])
    # A speed observation just before accelerometer recording start is valid;
    # the invariant is that every used observation is at or before the grid.
    assert np.all(np.nan_to_num(sensor_time-t[:,None],nan=0)<=0)
    name_id="lira_"+road.lower()+"_"+name.strip("/").replace("/","_").lower()
    metadata=dict(dataset="LiRA-CD v1",road=road,split=split,pass_name=name,unix_time_origin_s=origin,
        attribution="Skar et al. (2023), Live Road Assessment Custom Dataset, Technical University of Denmark; CC BY 4.0",
        source=provenance,alignment_policy=POLICY,alignment_audit=alignment,sections=sections,
        input_policy="Raw acc.xyz in g converted once to m/s²; last observed acceleration and speed on 100 Hz grid. Gyros missing. No filtering or future-value interpolation of inputs.",
        native_accel_median_interval_s=float(np.median(np.diff(acc[:,0]))),
        speed_key=speed_key,target_definition="Mean of 10 published left/right P79 IRI rows per 100-m section, aligned at acceleration acquisition time; gaps, slow or incomplete traversals masked.",
        disturbance_policy="Unknown everywhere: no event-level disturbance labels inferred from IRI, road distress tables or missing annotations.",
        limitations="GPS projection is approximate. Exact support convention of each 10-m IRI row is unspecified. Survey dates and wheel paths differ from car traversals. Repeated passes of the same road are correlated.")
    record=write_record(staging,dict(id=name_id,source="real",dataset="lira_cd",split=split,
        groups=["lira/"+road],original_recording=road+name,source_file=provenance["sensors"]["path"],
        source_sha256=provenance["sensors"]["sha256"]),x,mask,t,
        np.full((len(t),5),-100,np.int16),np.full(len(t),np.nan,np.float32),metadata)
    folder=staging/record["path"]
    for key,value in (("overall_iri",overall),("roughness_section",ids),("sensor_source_time",sensor_time)):
        path=folder/f"{key}.npy";np.save(path,value,allow_pickle=False)
        record["file_sha256"][path.name]=checksum(path)
    record["roughness_labeled_samples"]=int(np.isfinite(overall).sum())
    record["roughness_sections"]=len(sections)
    return record,dict(audit,status="prepared",samples=len(t),**alignment)


def prepare(root, source):
    root,source=Path(root).resolve(),Path(source).resolve()
    manifest_path=root/"manifest.json"
    before=manifest_path.read_bytes();manifest=json.loads(before)
    if any(r.get("dataset")=="lira_cd" for r in manifest["records"]):
        raise ValueError("LiRA is already integrated; use a fresh prepared root to rebuild")
    protocol=json.loads((source/"source_protocol.json").read_text())
    if protocol["road_family_splits"]!=ROADS:raise ValueError("LiRA road split policy changed")
    receipts={Path(r["path"]).name:r for r in json.loads((source/"download_manifest.json").read_text())["files"]}
    audit=[];records=[];inputs=[]
    # Fixed recipe and thresholds are recorded before opening any road,
    # including reserved M13. TEST is converted, never used for fitting.
    recipe_sha=checksum(Path(__file__))
    with tempfile.TemporaryDirectory(prefix=".lira_import_",dir=root) as temporary:
        staging=Path(temporary)
        for road,split in ROADS.items():
            iri,reference_gps,reference_hashes=load_reference(source,road,receipts)
            path=source/"raw/23192909"/f"{road}_HH.hdf5"
            sensor_hash=verify_input(path,receipts);inputs.extend([sensor_hash,*reference_hashes])
            with h5py.File(path,"r") as handle:
                for name in handle.attrs["GM_full_passes"]:
                    record,result=prepare_pass(staging,road,split,str(name),handle[name],iri,reference_gps,
                                               dict(sensors=sensor_hash,references=reference_hashes))
                    audit.append(result)
                    if record:records.append(record)
                    print(json.dumps({key:value for key,value in result.items() if key not in ["rejections"]}),flush=True)
        if not all(any(r["split"]==s and r["roughness_labeled_samples"] for r in records) for s in ROADS.values()):
            raise ValueError("A road split has no usable measured IRI")
        if any((root/r["path"]).exists() for r in records):raise ValueError("LiRA destination already exists")
        # Statistics can read both existing data and staging paths before commit.
        pending=[dict(r,path=str((staging/r["path"]).relative_to(root))) for r in records]
        combined=manifest["records"]+pending
        stats={s:training_statistics(root,combined,s) for s in ("real","synthetic","both")}
        by_real_dataset={}
        for collection in ("kaggle","lira"):
            chosen=[r for r in combined if r["source"]!="real" or (r.get("dataset")=="lira_cd")== (collection=="lira")]
            by_real_dataset[collection]={s:training_statistics(root,chosen,s,allow_missing_channels=True) for s in ("real","synthetic","both")}
        # Confirm no validation or test recording participated in any fit.
        train_ids={r["id"] for r in combined if r["split"]=="train"}
        for entry in [*stats.values(),*[v for options in by_real_dataset.values() for v in options.values()]]:
            if not set(entry["recording_ids"])<=train_ids:raise ValueError("Held-out normalizer input")
        manifest["records"]+=records
        manifest["train_statistics"]=stats
        manifest["train_statistics_by_real_dataset"]=by_real_dataset
        manifest["lira_cd_integration"]=dict(recipe_sha256=recipe_sha,policy=POLICY,road_splits=ROADS,
            source_protocol_sha256=checksum(source/"source_protocol.json"),inputs=inputs,pass_audits=audit,
            pretrained_or_evaluated_on_test=False,measurement="Independent P79 100-m section IRI",
            sensor_clock="100 Hz causal hold of raw approximately 50 Hz accelerometer; no added bandwidth",
            missing_channels=["gyro_x","gyro_y","gyro_z"],disturbance_labels="unknown",
            normalization="Only TRAIN observations; missing-only channels use neutral mean 0, std 1 in LiRA-only statistics",
            license="CC BY 4.0",attribution=protocol["attribution"],collection_url=protocol["collection_url"])
        manifest["summary"]={s:{split:dict(recordings=sum(r["source"]==s and r["split"]==split for r in manifest["records"]),
            seconds=sum(r["duration_seconds"] for r in manifest["records"] if r["source"]==s and r["split"]==split))
            for split in ("train","val","test")} for s in ("real","synthetic")}
        manifest["sources_policy"]+=" LiRA raw acceleration and speed with measured P79 IRI; fixed CPH1/M3/M13 road splits."
        manifest["excluded_sources"]="Measured-road calibration replays and non-admitted synthetic experiments remain excluded. LiRA missing/low-coverage passes are listed in lira_cd_integration."
        manifest.setdefault("overall_roughness",{})["real_labels"]="LiRA: measured P79 section IRI. Kaggle: unknown."
        if manifest_path.read_bytes()!=before:raise ValueError("Manifest changed during import")
        # Preserve the previous normalization/provenance for earlier experiments.
        snapshot=root/"manifest.before_lira.json"
        if snapshot.exists():raise ValueError("Pre-LiRA manifest snapshot already exists")
        snapshot.write_bytes(before)
        for r in records:(staging/r["path"]).rename(root/r["path"])
        temporary_manifest=root/"manifest.lira_pending.json"
        write_json(temporary_manifest,manifest);temporary_manifest.replace(manifest_path)
    summary={s:dict(recordings=sum(r["split"]==s for r in records),
        hours=sum(r["duration_seconds"] for r in records if r["split"]==s)/3600,
        sections=sum(r["roughness_sections"] for r in records if r["split"]==s),
        labeled_samples=sum(r["roughness_labeled_samples"] for r in records if r["split"]==s)) for s in ROADS.values()}
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[2]/"data/road_quality_reference")
    args=parser.parse_args();prepare(args.data_root,args.source_root)
