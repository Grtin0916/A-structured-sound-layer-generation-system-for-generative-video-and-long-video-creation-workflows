"""Stable A/B/C/D experiment contracts."""
from soundlayer.runner.contracts import digest_value
STRATEGIES=("A_CONTROL","B_DSS_SCHEDULE","C_AVAILABILITY_RERANK","D_TARGETED_REPAIR")
def build_contracts(cases,config_digest):
    rows=[]
    for case_order,case in enumerate(cases):
        for strategy_order,strategy in enumerate(STRATEGIES):
            row={"case_id":case["matrix_case_id"],"strategy_id":strategy,"case_order":case_order,
                 "strategy_order":strategy_order,"source_group":"CONTROL" if strategy=="A_CONTROL" else "CONTROL_TRANSFORM",
                 "live_group_available":False,"metric_available":strategy!="A_CONTROL" or True,
                 "unavailable_reason":"LIVE_RUNTIME_BLOCKED","publish_decision":
                 "BLOCKED" if case["repair_decision"]=="REPAIR_REJECTED" else "PROVISIONAL_SELECTED",
                 "claim_boundary":{"human_preference_proven":False,"final_selected":False}}
            row["contract_key"]=digest_value({"case_id":row["case_id"],"strategy":strategy,"config":config_digest})
            rows.append(row)
    return rows
