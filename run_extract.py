import sys
from loguru import logger
from omegaconf import DictConfig

from reqap.library.library import load_config, set_random_seed
from reqap.extract.extract_module import ExtractModule
set_random_seed()


def derive_attributes(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.derive_attributes(config)


def derive_extract_call_attributes(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.derive_extract_call_attributes(config)


def derive_attribute_mappings(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.derive_attribute_mappings(config)


def derive_data(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.derive_data(config)


def train(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.train()


def evaluate(config: DictConfig):
    extract = ExtractModule(config.extract)
    extract.evaluate()


if __name__ == "__main__":
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python run_extract.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/reqap.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # run
    function = sys.argv[1]
    if function.startswith("--derive_attributes"):
        derive_attributes(config)
    elif function.startswith("--derive_extract_call_attributes"):
        derive_extract_call_attributes(config)
    elif function.startswith("--derive_attribute_mappings"):
        derive_attribute_mappings(config)
    elif function.startswith("--derive_data"):
        derive_data(config)
    elif function.startswith("--train"):
        train(config)
    elif function.startswith("--eval"):
        evaluate(config)
    else:
        raise Exception(f"Unknown function {function}.")
