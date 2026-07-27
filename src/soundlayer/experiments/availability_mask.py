"""Availability is a hard mask, never a synthetic zero score."""
def available(record,root):
    from pathlib import Path
    if record.get("status")!="SUCCEEDED":return False,"STATUS_"+record.get("status","UNKNOWN")
    if not record.get("output_path"):return False,"ARTIFACT_MISSING"
    if not (Path(root)/record["output_path"]).is_file():return False,"ARTIFACT_MISSING"
    if record.get("repair_decision")=="REPAIR_REJECTED":return False,"REPAIR_REJECTED"
    return True,""
def available_values(records,field):
    return [x[field] for x in records if x.get(field) is not None and x.get("metric_available",False)]
