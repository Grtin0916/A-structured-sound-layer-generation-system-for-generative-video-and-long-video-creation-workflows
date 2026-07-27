"""Hard-gate availability before proxy ranking."""
from soundlayer.experiments.availability_mask import available
def select(candidates,root):
    feasible=[];rejected=[]
    for item in candidates:
        ok,reason=available(item,root)
        if ok:feasible.append(item)
        else:rejected.append({"artifact":item.get("output_path",""),"reason":reason})
    if not feasible:return None,{"blocked_reason":"NO_AVAILABLE_CANDIDATE","rejected":rejected}
    feasible.sort(key=lambda x:(-x["proxy_score"],x["estimated_edit_cost"],x["output_path"]))
    return feasible[0],{"why_selected":"availability gates passed, then proxy score and edit cost","rejected":rejected}
