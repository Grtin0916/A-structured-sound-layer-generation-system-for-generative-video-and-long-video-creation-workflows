# Week17 True-aware Demo Pack Seed

## What this demo shows

This seed demonstrates one claim-safe true MMAudio video-conditioned candidate flowing through the three-repo system.

## System path

1. Mainbase produced one true MMAudio candidate for `glass_drop_room_001`.
2. Mainbase wrapped it in a claim-safe result card.
3. Java exposed the result through an artifact-backed API.
4. Cloud consumed the Java report and generated a demo gate seed.
5. Cloud emitted Prometheus metrics and a Grafana dashboard seed.

## Claim boundary

Allowed:

- One true MMAudio video-conditioned candidate exists.
- Java can expose this result as a result-card API.
- Cloud can treat it as ready for Friday demo packaging.

Forbidden:

- Do not claim true MMAudio batch success.
- Do not claim full 28-candidate ranking.
- Do not claim production SLO.
- Do not claim k6 threshold pass.

## Primary audio

`audio/glass_drop_room_001__mmaudio__true_replacement_v0.flac`

## Key numbers

- safe true MMAudio count: `1`
- raw candidate context count: `9`
- ready for Friday demo pack: `True`
