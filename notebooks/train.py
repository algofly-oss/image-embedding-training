import hydra
from omegaconf import DictConfig
from oml.const import HYDRA_BEHAVIOUR
from oml.lightning.pipelines.train import extractor_training_pipeline
import os

# os.environ['WANDB_DISABLED'] = 'true'
# os.environ['WANDB_API_KEY'] = 'xxxx'

@hydra.main(config_path=".", config_name="train.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    extractor_training_pipeline(cfg)

main_hydra()