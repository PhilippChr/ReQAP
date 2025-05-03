import sys
import json
from loguru import logger
from omegaconf import DictConfig

from reqap.library.library import load_json, load_config, set_random_seed
from reqap.qu.qu_module import QuestionUnderstandingModule
set_random_seed()


def derive_data(config: DictConfig):
    qu = QuestionUnderstandingModule(config.qu, train=True)  # avoid loading QU model
    qu.derive_training_data()


def train(config: DictConfig):
    qu = QuestionUnderstandingModule(config.qu, train=True)
    qu.train()


def evaluate(config: DictConfig):
    qu = QuestionUnderstandingModule(config.qu, train=False)
    qu.evaluate()


def process_queries(qu_supervisor, queries_path, output_path):
    data = load_json(queries_path)

    processed_data = list()
    for instance in data:
        question = instance["question"]
        print(f"\n\nQuestion: {question}")
        qu_plans = qu_supervisor.run(question)
        print(f"QU plans: {qu_plans}")
        processed_data.append({
            "question": question,
            "sql": instance["sql"],
            "qu_plans": [qu_plan.to_dict() for qu_plan in qu_plans]
        })
    with open(output_path, "w") as fp:
        fp.write(json.dumps(processed_data, indent=4))


def main():
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python run_qu.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/reqap.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # run
    function = sys.argv[1]
    if function.startswith("--derive_data"):
        derive_data(config)
    elif function.startswith("--train"):
        train(config)
    elif function.startswith("--eval"):
        evaluate(config)


if __name__ == "__main__":
    main()
