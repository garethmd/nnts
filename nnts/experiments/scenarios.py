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
    skip: int = 0

    def copy(self):
        return CovariateScenario(
            prediction_length=self.prediction_length,
            error=self.error,
            conts=self.conts.copy(),
            pearson=self.pearson,
            noise=self.noise,
            covariates=self.covariates,
            seed=self.seed,
            skip=self.skip,
        )

    @property
    def name(self):
        if self.skip == 1:
            return f"cov-{self.covariates}-pearsn-{str(round(self.pearson, 2))}-pl-{str(self.prediction_length)}-seed-{self.seed}-skip-{self.skip}"
        return f"cov-{self.covariates}-pearsn-{str(round(self.pearson, 2))}-pl-{str(self.prediction_length)}-seed-{self.seed}"
