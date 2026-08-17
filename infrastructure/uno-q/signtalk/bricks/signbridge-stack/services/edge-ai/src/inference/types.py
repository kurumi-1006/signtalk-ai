from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class Prediction:
    label: str
    text: str
    confidence: float
    margin: float
    landmark_coverage: float
    top_k: tuple[Candidate, ...]
    diagnostics: dict[str, object]
    accepted: bool | None = None
