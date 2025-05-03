import sys
import json
from omegaconf import DictConfig
from loguru import logger

from reqap.library.library import get_persona_names, load_config, avg, load_json, store_json
from reqap.library.metrics import hit_at_1


def evaluate(config: DictConfig, split: str="test"):
    # init paths
    persona_dir = f"{config.benchmark.benchmark_dir}/{split}"
    result_dir = config.benchmark.result_dir
    personas = get_persona_names(persona_dir)
    
    # init global results
    all_hits_at_1_list = list()
    all_hits_at_1_10_list = list()
    all_hits_at_1_20_list = list()
    all_persona_results = ""
    
    # iterate through personas
    for i, persona in enumerate(personas):
        # load data
        input_path = f"{result_dir}/{persona}/result.jsonl"
        questions_path = f"{persona_dir}/{persona}/questions.json"
        logger.debug(f"Loading result from {input_path}...")
        questions = load_json(questions_path)
        
        # init result lists
        hit_at_1_list = list()
        relaxed_hit_at_1_10_list = list()
        relaxed_hit_at_1_20_list = list()
        num_failures = 0
        
        # process data
        c = 0
        with open(input_path, "r") as fp:
            for j, line in enumerate(fp):
                instance = json.loads(line)
                derived_answer = instance["derived_answer"]
                
                # assert that questions are as expected
                if not questions[c]["question"] == instance["question"]:
                    c += 1
                assert questions[c]["question"] == instance["question"],\
                    f'Failure with questions: questions[j]["question"]={questions[j]["question"]}, instance["question"]={instance["question"]}'
                gold_answers = questions[c]["answers"]
                
                # process system failure
                if instance.get("failed", False):
                    hit_at_1_list.append(0.0)
                    relaxed_hit_at_1_10_list.append(0.0)
                    relaxed_hit_at_1_20_list.append(0.0)
                    num_failures += 1
                    continue
            
                # compute metrics and append results
                hit_at_1_score = hit_at_1(derived_answer, gold_answers)
                hit_at_1_10_score = hit_at_1(derived_answer, gold_answers, relax_factor=0.1)
                hit_at_1_20_score = hit_at_1(derived_answer, gold_answers, relax_factor=0.2)
                hit_at_1_list.append(hit_at_1_score)
                relaxed_hit_at_1_10_list.append(hit_at_1_10_score)
                relaxed_hit_at_1_20_list.append(hit_at_1_20_score)
                c += 1
        
        # add to list for all personas
        all_hits_at_1_list += hit_at_1_list
        all_hits_at_1_10_list += relaxed_hit_at_1_10_list
        all_hits_at_1_20_list += relaxed_hit_at_1_20_list

        # log current results
        logger.info(f"Persona {i+1}: avg(hit_at_1_list)={avg(hit_at_1_list)} ({sum(hit_at_1_list)}/{len(hit_at_1_list)})")
        logger.info(f"Persona {i+1}: avg(relaxed_hit_at_1_10_list)={avg(relaxed_hit_at_1_10_list)} ({sum(relaxed_hit_at_1_10_list)}/{len(relaxed_hit_at_1_10_list)})")
        logger.info(f"Persona {i+1}: avg(relaxed_hit_at_1_20_list)={avg(relaxed_hit_at_1_20_list)} ({sum(relaxed_hit_at_1_20_list)}/{len(relaxed_hit_at_1_20_list)})")
        logger.info(f"Persona {i+1}: num failures={num_failures} ({num_failures}/{len(hit_at_1_list)})")
        all_persona_results += f"{round(avg(hit_at_1_list), 3)}, {round(avg(relaxed_hit_at_1_10_list), 3)}, {round(avg(relaxed_hit_at_1_20_list), 3)}, "

    # write out per-question result for statistical significance testing
    res_lists_path = f"{result_dir}/result_lists.json"
    store_json(res_lists_path, {"hit_at_1": all_hits_at_1_list, "relaxed_hit_at_1": all_hits_at_1_10_list, "relaxed_hit_at_1_20": all_hits_at_1_20_list})
    
    # log result for Google sheet
    all_persona_results = f"{round(avg(all_hits_at_1_list), 3)}, {round(avg(all_hits_at_1_10_list), 3)}, {round(avg(all_hits_at_1_20_list), 3)}, " + all_persona_results[:-2]
    logger.info(f"Result (for Google Sheet): {all_persona_results}")


def main():
    config_path = "config/perqa/reqap_openai.yml" if len(sys.argv) < 2 else sys.argv[1]
    split = "test" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)
    evaluate(config, split)


if __name__ == "__main__":
    main()
