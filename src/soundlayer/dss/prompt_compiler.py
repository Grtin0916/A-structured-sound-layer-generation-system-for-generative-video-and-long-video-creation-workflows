from typing import Dict, List

from soundlayer.dss.schema import DSSCase, DSSEvent


def _event_line(event: DSSEvent) -> str:
    return (
        f"{event.time_sec:.2f}s: object={event.object}; "
        f"action={event.action}; sound={event.sound_intent}; "
        f"layer={event.layer}; priority={event.priority}; tolerance={event.tolerance_ms}ms"
    )


def _plain_event_line(event: DSSEvent) -> str:
    return f"around {event.time_sec:.2f}s, {event.object} {event.action} with {event.sound_intent}"


def _join_events(case: DSSCase) -> str:
    events = sorted(case.events, key=lambda item: item.time_sec)
    if not events:
        return "No explicit events were provided; infer only visible physical actions."
    return " | ".join(_event_line(event) for event in events)


def _join_plain_events(case: DSSCase) -> str:
    events = sorted(case.events, key=lambda item: item.time_sec)
    if not events:
        return "infer the main visible physical actions"
    return "; ".join(_plain_event_line(event) for event in events)


def compile_prompt(case: DSSCase, variant: str) -> Dict[str, object]:
    avoid_text = ", ".join(case.avoid) if case.avoid else "speech, voiceover, lyrics, unrelated music"
    event_text = _join_events(case)
    plain_event_text = _join_plain_events(case)

    prompts = {
        "naive": (
            f"Create realistic sound effects for this short video. "
            f"Scene: {case.scene}. "
            f"Use natural ambience and match visible actions. "
            f"Avoid {avoid_text}."
        ),
        "naive_rich": (
            f"Create realistic sound effects for this short video. "
            f"Scene: {case.scene}. "
            f"The visible actions include: {plain_event_text}. "
            f"Make the sounds natural, synchronized, and cinematic. "
            f"Avoid {avoid_text}."
        ),
        "dss_global": (
            f"Generate synchronized audio for a {case.duration_sec:.1f}s video. "
            f"Global scene: {case.scene}. "
            f"Overall style: {case.style}. "
            f"Keep ambience stable, keep foley clear, avoid forbidden content: {avoid_text}."
        ),
        "dss_event_timeline": (
            f"Generate audio for a {case.duration_sec:.1f}s video using this event timeline. "
            f"Scene: {case.scene}. "
            f"Event timeline: {event_text}. "
            f"Preserve temporal order. Align transient sounds to event timestamps. "
            f"Do not add: {avoid_text}."
        ),
        "dss_layer_avoid": (
            f"Generate layered video-synchronized audio. "
            f"Scene layer: {case.scene}. "
            f"Foley layer must emphasize these events: {event_text}. "
            f"Ambience layer should remain low, continuous, and non-dominant. "
            f"Music layer should be absent unless explicitly required. "
            f"Forbidden content: {avoid_text}. "
            f"No speech, no lyrics, no unrelated background music."
        ),
    }

    if variant not in prompts:
        raise ValueError(f"unsupported_variant={variant}")

    prompt = prompts[variant]
    validation_errors = case.validate()

    return {
        "case_id": case.case_id,
        "variant": variant,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "event_count": len(case.events),
        "avoid_count": len(case.avoid),
        "duration_sec": case.duration_sec,
        "validation_errors": validation_errors,
        "ready_for_generation": len(validation_errors) == 0,
    }


def supported_variants() -> List[str]:
    return ["naive", "naive_rich", "dss_global", "dss_event_timeline", "dss_layer_avoid"]
