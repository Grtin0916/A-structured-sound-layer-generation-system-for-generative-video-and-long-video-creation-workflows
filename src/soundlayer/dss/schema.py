from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _first_present(raw: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, "", []):
            return raw[key]
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


@dataclass
class DSSEvent:
    time_sec: float
    object: str
    action: str
    sound_intent: str
    priority: int = 3
    tolerance_ms: int = 500
    layer: str = "foley"

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "DSSEvent":
        time_value = _first_present(
            raw,
            ["time_sec", "time", "start_sec", "timestamp_sec", "event_time_sec", "onset_sec"],
            0.0,
        )
        object_value = _first_present(
            raw,
            ["object", "actor", "target", "entity", "source_object"],
            "visible object",
        )
        action_value = _first_present(
            raw,
            ["action", "event", "verb", "motion", "description"],
            "visible action",
        )
        sound_value = _first_present(
            raw,
            ["sound_intent", "sound", "audio", "sound_description", "expected_sound", "description"],
            "natural synchronized sound",
        )
        priority_value = _first_present(raw, ["priority", "importance", "weight"], 3)
        tolerance_value = _first_present(raw, ["tolerance_ms", "tolerance", "sync_tolerance_ms"], 500)
        layer_value = _first_present(raw, ["layer", "role", "audio_layer"], "foley")

        return DSSEvent(
            time_sec=_to_float(time_value, 0.0),
            object=str(object_value),
            action=str(action_value),
            sound_intent=str(sound_value),
            priority=_to_int(priority_value, 3),
            tolerance_ms=_to_int(tolerance_value, 500),
            layer=str(layer_value),
        )


@dataclass
class DSSCase:
    case_id: str
    duration_sec: float
    scene: str
    events: List[DSSEvent] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    style: str = "cinematic realistic foley, clean ambience, no speech"
    source_file: Optional[str] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any], source_file: Optional[str] = None) -> "DSSCase":
        video = raw.get("video", {})
        if not isinstance(video, dict):
            video = {}

        constraints = raw.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        layers = raw.get("layers", {})
        if not isinstance(layers, dict):
            layers = {}

        case_id = _first_present(raw, ["case_id", "id", "name", "case"], "unknown_case")
        duration_value = _first_present(
            raw,
            ["duration_sec", "duration", "video_duration_sec"],
            _first_present(video, ["duration_sec", "duration"], 10.0),
        )
        scene_value = _first_present(
            raw,
            ["scene", "scene_description", "description", "baseline_prompt", "prompt"],
            "short video scene",
        )

        events_raw = _first_present(
            raw,
            ["events", "expected_events", "event_timeline", "timeline"],
            [],
        )
        events: List[DSSEvent] = []
        if isinstance(events_raw, list):
            for item in events_raw:
                if isinstance(item, dict):
                    events.append(DSSEvent.from_dict(item))

        avoid_raw = _first_present(
            raw,
            ["avoid", "negative_prompt", "forbidden", "avoid_list"],
            _first_present(constraints, ["avoid", "negative_prompt", "forbidden"], None),
        )
        if avoid_raw is None:
            avoid_raw = ["speech", "voiceover", "lyrics", "unrelated music"]
        if isinstance(avoid_raw, str):
            avoid = [x.strip() for x in avoid_raw.replace(";", ",").split(",") if x.strip()]
        elif isinstance(avoid_raw, list):
            avoid = [str(x).strip() for x in avoid_raw if str(x).strip()]
        else:
            avoid = ["speech", "voiceover", "lyrics", "unrelated music"]

        style = str(
            _first_present(
                raw,
                ["style", "audio_style"],
                _first_present(constraints, ["style"], "cinematic realistic foley, clean ambience, no speech"),
            )
        )

        # If layer config explicitly disables speech/music, keep that visible in avoid list.
        for forbidden in ["speech", "lyrics", "voiceover"]:
            if forbidden not in [x.lower() for x in avoid]:
                avoid.append(forbidden)

        return DSSCase(
            case_id=str(case_id),
            duration_sec=_to_float(duration_value, 10.0),
            scene=str(scene_value),
            events=events,
            avoid=avoid,
            style=style,
            source_file=source_file,
        )

    def validate(self) -> List[str]:
        errors: List[str] = []

        if not self.case_id or self.case_id == "unknown_case":
            errors.append("missing_case_id")

        if self.duration_sec <= 0:
            errors.append("invalid_duration_sec")

        if not self.scene or self.scene == "short video scene":
            errors.append("weak_or_missing_scene")

        if len(self.events) < 1:
            errors.append("missing_events")

        for idx, event in enumerate(self.events):
            if event.time_sec < 0:
                errors.append(f"event_{idx}_negative_time")
            if event.time_sec > self.duration_sec + 1.0:
                errors.append(f"event_{idx}_time_out_of_range")
            if not event.sound_intent:
                errors.append(f"event_{idx}_missing_sound_intent")

        return errors
