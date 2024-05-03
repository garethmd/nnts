from dataclasses import dataclass, field


@dataclass
class BaseScenario:
    prediction_length: int
    conts: list = field(default_factory=list)
    seed: int = 42

    def copy(self):
        return self.__class__(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
        )


@dataclass
class Scenario(BaseScenario):
    covariates: int = field(init=False)

    def __post_init__(self):
        self.covariates = len(self.conts)

    def copy(self):
        return Scenario(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
        )

    @property
    def name(self):
        return (
            f"cov-{self.covariates}-pl-{str(self.prediction_length)}-seed-{self.seed}"
        )


@dataclass
class CovariateScenario(BaseScenario):
    error: int = 0
    pearson: float = 0
    noise: float = 0
    covariates: int = 0
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
