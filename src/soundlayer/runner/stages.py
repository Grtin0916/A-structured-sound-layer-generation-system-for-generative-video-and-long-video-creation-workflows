"""Replay stage definitions and artifact selection."""
STAGES=[
 {"stage_id":"inventory","dependencies":[]},
 {"stage_id":"preflight","dependencies":["inventory"]},
 {"stage_id":"resolve_case","dependencies":["preflight"]},
 {"stage_id":"compile_dss","dependencies":["resolve_case"]},
 {"stage_id":"acquire_candidates","dependencies":["compile_dss"]},
 {"stage_id":"evaluate","dependencies":["acquire_candidates"]},
 {"stage_id":"rerank","dependencies":["evaluate"]},
 {"stage_id":"repair","dependencies":["rerank"]},
 {"stage_id":"mix","dependencies":["repair"],"exclusive_outputs":["published_mix"]},
 {"stage_id":"report","dependencies":["mix"],"exclusive_outputs":["run_report"]},
 {"stage_id":"publish","dependencies":["report"],"exclusive_outputs":["publish_status"]},
]
CACHEABLE={"inventory","preflight","resolve_case","compile_dss","acquire_candidates","evaluate","rerank","repair"}
