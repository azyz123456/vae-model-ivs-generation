from dataclasses import dataclass

@dataclass
class TrainCfg:
    beta: float = 1.0
    lambda_cal: float = 0.0
    lambda_bfly: float = 0.0
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    device: str = "cuda"  
    verbose: bool = True


@dataclass
class PosteriorCfg:
    obs_noise: float = 0.02
    n_samples: int = 200
    burn_in: int = 200
    thin: int = 2
    step_size: float = 1e-3
    device: str = "cuda"  
    seed: int = 123