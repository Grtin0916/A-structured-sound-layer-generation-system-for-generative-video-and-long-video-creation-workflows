"""Deterministic DSS scheduling over PCM16 Control audio."""
import json,math,struct,wave
from pathlib import Path
from .event_envelope import smooth_weight
def read_pcm(path):
    with wave.open(str(path),"rb") as w:
        meta=(w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes());raw=w.readframes(meta[3])
    if meta[1]!=2:raise ValueError("PCM16 required")
    return meta,list(struct.unpack("<"+"h"*(len(raw)//2),raw))
def write_pcm(path,meta,samples):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),"wb") as w:
        w.setnchannels(meta[0]);w.setsampwidth(meta[1]);w.setframerate(meta[2]);w.writeframes(struct.pack("<"+"h"*len(samples),*samples))
def compile_control(source,dss_path,output,priority_gain,ceiling_dbfs=-1.0):
    meta,samples=read_pcm(source);channels,_,rate,frames=meta;dss=json.loads(Path(dss_path).read_text())
    gains=[1.0]*frames;windows=[]
    for event in dss["events"]:
        start=max(0,min(frames,int(event["time_s"]*rate)));end=max(start,min(frames,int((event["time_s"]+event["duration_s"])*rate)))
        if end<=start:continue
        gain=float(priority_gain[str(event["priority"])]);fade=min(int(rate*.02),(end-start)//2)
        for i in range(start,end):
            weight=smooth_weight(i,start,end,fade);gains[i]*=1+(gain-1)*weight
        windows.append({"event_id":event["event_id"],"start_frame":start,"end_frame":end,"priority":event["priority"]})
    ceiling=int(32767*(10**(ceiling_dbfs/20)));out=[]
    for frame in range(frames):
        for c in range(channels):out.append(max(-ceiling,min(ceiling,round(samples[frame*channels+c]*gains[frame]))))
    write_pcm(output,meta,out)
    before=sum(x*x for x in samples)/max(1,len(samples));after=sum(x*x for x in out)/max(1,len(out))
    return {"duration_match":True,"clip_ratio":sum(abs(x)>=32767 for x in out)/max(1,len(out)),
            "rms_before":math.sqrt(before)/32768,"rms_after":math.sqrt(after)/32768,
            "proxy_score":math.sqrt(after)/32768,"windows":windows,"avoid_constraints":sorted({v for e in dss["events"] for v in e.get("avoid",[])})}
