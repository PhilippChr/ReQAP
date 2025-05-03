import sys
from loguru import logger
from omegaconf import DictConfig

from reqap.library.library import get_persona_names, load_config, set_random_seed
from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import IndexConstructor
from reqap.retrieval.crossencoder.crossencoder_module import CrossEncoder
set_random_seed()


def construct_index(config: DictConfig):
    # init
    benchmark_dir = config.benchmark.benchmark_dir
    splade_model = Splade(config.splade.splade_model_type_or_path, agg="max")
    
    # process splits
    for split in ["train", "dev", "test"]:
        logger.info(f"Starting with index construction on {split} set.")
        persona_dir = f"{benchmark_dir}/{split}"
        personas = get_persona_names(persona_dir)
        for persona in personas:
            splade_persona_config = config.splade.copy()
            splade_index_dir = config.splade.splade_indices_dir
            splade_persona_config["obs_events_csv_path"] = f"{persona_dir}/{persona}/{persona}_obs.csv"
            splade_persona_config["splade_index_path"] = f"{splade_index_dir}/{persona}.splade_index"
            construct_index_for_persona(splade_persona_config, splade_model)
            logger.info(f"...done with constructing index for {persona}.")


def construct_index_for_persona(splade_persona_config: DictConfig, splade_model: Splade):
    constructor = IndexConstructor()
    constructor.run(splade_persona_config, splade_model)


def derive_retrieve_calls(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.derive_retrieve_calls()


def derive_data(config: DictConfig, persona: str=None):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.derive_data(persona)


def derive_equivalent_retrieve_queries(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.derive_equivalent_retrieve_queries()


def train_ce_events(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.train_ce_events()


def train_ce_patterns(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.train_ce_patterns()


def evaluate_ce_events(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.evaluate_ce_events()


def evaluate_ce_patterns(config: DictConfig):
    ce = CrossEncoder(config=config, ce_config=config.crossencoder)
    ce.evaluate_ce_patterns()


def dev(config: DictConfig):
    from reqap.retrieval.retrieval import Retrieval
    from reqap.qu.qu_operators import RETRIEVE
    retrieval = Retrieval(config)
    res = RETRIEVE(retrieval, query="I listened to music")
    logger.info(f"Result: {res}")


def main():
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python run_retrieval.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/reqap_sft.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # run
    function = sys.argv[1]
    if function.startswith("--construct_index"):
        construct_index(config)
    elif function.startswith("--derive_retrieve_calls"):
        derive_retrieve_calls(config)
    elif function.startswith("--derive_data"):
        persona = None if len(sys.argv) < 4 else sys.argv[3]  # enable providing specific person here
        derive_data(config, persona)
    elif function.startswith("--derive_equivalent_retrieve_queries"):
        derive_equivalent_retrieve_queries(config)
    elif function.startswith("--train_ce_events"):
        train_ce_events(config)
    elif function.startswith("--train_ce_patterns"):
        train_ce_patterns(config)
    elif function.startswith("--eval_ce_events"):
        evaluate_ce_events(config)
    elif function.startswith("--eval_ce_patterns"):
        evaluate_ce_patterns(config)
    elif function.startswith("--dev"):
        dev(config)
    else:
        raise Exception(f"Unknown function {function}.")


if __name__ == "__main__":
    main()
