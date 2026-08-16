from dataclasses import dataclass
@dataclass(frozen=True)
class Prediction:
    label: str
    text: str
    confidence: float
class MockPredictor:
    def predict(self) -> Prediction: return Prediction(label='hello', text='Xin chào', confidence=0.95)
