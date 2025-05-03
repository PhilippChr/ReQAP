import sys
from loguru import logger

from reqap.reqap import ReQAP
from reqap.library.library import load_config, set_random_seed
set_random_seed()


def main():
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python pipeline.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/reqap.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # create ReQAP instance
    reqap = ReQAP(config)
    logger.info("ReQAP loaded.")

    # run
    function = sys.argv[1]
    if function.startswith("--qud"):
        split = function.replace("--qud", "")
        split = split.replace("_", "").replace("-", "")
        split = split if split else "test"
        reqap.run_qud_on_split(split)

    elif function.startswith("--otx"):
        split = function.replace("--otx", "")
        split = split.replace("_", "").replace("-", "")
        split = split if split else "test"
        reqap.run_otx_on_split(split)

    elif function.startswith("--dev"):
        reqap.run_icl_examples()

    elif function.startswith("--loop"):
        split = function.replace("--loop", "")
        split = split.replace("_", "").replace("-", "")
        split = split if split else "train"
        loop_persona = None if len(sys.argv) < 4 else sys.argv[3]
        reqap.training_loop(split, loop_persona=loop_persona)

    elif function.startswith("--example"):
        reqap.example("how often did I listen to music last month?")
        
    else:
        raise Exception(f"Unknown function {function}.")


if __name__ == "__main__":
    main()