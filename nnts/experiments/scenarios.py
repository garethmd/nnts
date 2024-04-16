from dataclasses import dataclass, field


@dataclass
class CovariateScenario:
    prediction_length: int
    error: int
    conts: list = field(default_factory=list)
    pearson: float = 0
    noise: float = 0
    covariates: int = 0
    seed: int = 42

    @property
    def name(self):
        return f"cov-{self.covariates}-pearsn-{str(round(self.pearson, 2))}-pl-{str(self.prediction_length)}-seed-{self.seed}"
