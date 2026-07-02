# Week17 True-aware Demo Pack Export

## Open locally

Run from the exported directory:

    cd week17_true_aware_demo_pack_seed
    python -m http.server 8787

Then open this URL in a browser:

    http://127.0.0.1:8787/index.html

## What this demonstrates

- One claim-safe true MMAudio video-conditioned candidate.
- Mainbase result-card bridge.
- Java artifact-backed result-card API evidence.
- Cloud demo gate seed.
- Prometheus metrics sample and Grafana dashboard seed.

## Case

- case_id: glass_drop_room_001
- primary model: MMAudio
- safe true MMAudio count: 1
- raw candidate context count: 9
- primary audio: audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac

## Claim boundary

Allowed:

- One true MMAudio video-conditioned candidate is available.
- Java can expose this result as an artifact-backed result-card API.
- Cloud can use this as a Friday demo gate seed.

Forbidden:

- No true MMAudio batch success.
- No full candidate ranking claim.
- No production SLO claim.
- No k6 threshold pass claim.
