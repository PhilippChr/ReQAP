import os
import sys
import torch
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Dict

from reqap.library.metrics import hit_at_1
from reqap.library.library import load_config, load_json, load_jsonl, store_jsonl, get_persona_names, load_prompt_template, handle_output_file, load_txt, set_random_seed
from reqap.llm.seq2seq_model import DatasetSeq2Seq, Seq2SeqModel
from reqap.llm.causal_model import DatasetCausalModel, CausalModel
from reqap.retrieval.query_execution import QueryExecution
from reqap.qu.qu_module import QuestionUnderstandingModule
set_random_seed()

class QueryGeneration:
    ICL_PROMPT_TEMPLATE = (
        "Question: {question}"
    )

    OUTPUT_TEMPLATE = "{sql_query}"
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.instruction = load_txt(config.query_generation.instruction)
        self.prompt = load_prompt_template(config.query_generation.prompt)
        self.icl_examples = load_json(config.query_generation.icl_examples)
        self.sql_schema = load_txt(config.query_generation.sql_schema)
        self.model_loaded = False

    def run(self, split: str="test", override: bool=False):
        # init
        self.load_model()
        benchmark_dir = self.config.benchmark.benchmark_dir
        result_dir = self.config.benchmark.result_dir
        persona_dir = os.path.join(benchmark_dir, split)
        personas = get_persona_names(persona_dir)

        # load existing result or start from scratch if override=True
        result_path = self.config.query_generation.result_path
        question_to_sql_query = QuestionUnderstandingModule.initialize_question_to_qu_plan(result_path, override)

        # iterate through personas
        for i, persona in enumerate(personas):
            # init paths for persona
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            questions_path = f"{persona_dir}/{persona}/questions.json"
            output_path = f"{result_dir}/{persona}/result.jsonl"
            data = load_json(questions_path)
            data = handle_output_file(output_path, data, override)

            # init query execution
            query_execution = QueryExecution(obs_events_csv_path=obs_events_csv_path)

            # process questions
            logger.debug(f"Loaded questions from {questions_path}")
            for j, instance in tqdm(enumerate(data), total=len(data)):
                logger.debug(f"Starting with inference for question {j} / {len(data)}, persona {i} / {len(personas)}")
                question = instance["question"]
                reference_date = instance["reference_date"]

                # generate SQL query
                if question in question_to_sql_query:
                    sql_query_output = question_to_sql_query[question]
                else:
                    logger.debug(f"Running model for question={question}")
                    sql_query_output = self.generate_sql_query(question)
                    store_jsonl(result_path, [{question: sql_query_output}], file_mode="a")
                    question_to_sql_query[question] = sql_query_output
                sql_query = self.format_sql_query(sql_query_output)

                # run SQL query
                sql_query = query_execution.adjust_sql_query_reference_data(sql_query, reference_date)
                instance["failed"] = False
                try:
                    res_df = query_execution.db.query(sql_query).df()
                    answer = query_execution.parse_query_result(res_df, json_serializable=True)
                except Exception as e:
                    answer = None
                    logger.warning(f"Error with sql_query={sql_query}: {e}")
                    instance["failed"] = True
                    instance["error"] = str(e)
            
                instance["generated_sql_query_output"] = sql_query_output
                instance["generated_sql_query"] = sql_query
                instance["derived_answer"] = answer
                store_jsonl(output_path, [instance], file_mode="a")

    @staticmethod
    def format_sql_query(sql_query: str):
        """ Remove any LLM formatting from SQL query. """
        sql_query = sql_query.replace("`", "")
        sql_query = sql_query.replace("sql", "")
        return sql_query

    def generate_sql_query(self, question: str) -> str:
        if isinstance(self.model, Seq2SeqModel):
            sql_query = self.model.single_inference(input_text=question, generation_params=self.config.query_generation.generation_params)
            logger.debug(f"SQL query: {sql_query}")
        elif isinstance(self.model, CausalModel):
            input_str = self.represent_input_causal(question=question, tokenizer=self.model.tokenizer)
            sql_query = self.model.inference(input_text=input_str, sampling_params=self.config.query_generation.sampling_params)
            logger.debug(f"SQL query: {sql_query}")
        else:
            prompt = self.prompt.render(
                question=question,
                sql_schema=self.sql_schema
            )
            dialog = self.construct_dialog(
                prompt=prompt
            )
            sql_query = self.model.batch_inference(
                [dialog],
                sampling_params=self.config.query_generation.sampling_params
            )[0]  # use batch_inference to make use of cache
            logger.debug(f"SQL query: {sql_query}")
        return sql_query

    def construct_dialog(self, prompt: str) -> List[Dict]:
        dialog = [{
            "role": "system",
            "content": self.instruction,
        }]
        for ex in self.icl_examples:
            dialog.append({
                "role": "user",
                "content": self.ICL_PROMPT_TEMPLATE.format(**ex),
            })
            dialog.append({
                "role": "assistant",
                "content": self.OUTPUT_TEMPLATE.format(**ex),
        })
        dialog += [{
            "role": "user",
            "content": prompt,
        }]
        return dialog

    def load_model(self) -> None:
        """
        Initiate the query generation module.
        """
        if self.model_loaded:
            return
        mode = self.config.query_generation.mode
        if mode == "openai":
            from reqap.llm.openai import OpenAIModel
            model = OpenAIModel(openai_config=self.config.openai, use_cache=self.config.openai.use_cache)
        elif mode == "instruct_model":
            from reqap.llm.instruct_model import InstructModel
            model = InstructModel(instruct_config=self.config.instruct_model, use_cache=True)
        elif mode == "seq2seq":
            model = Seq2SeqModel(seq2seq_config=self.config.query_generation, train=False)
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda"))
        elif mode == "causal":
            model = CausalModel(self.config.query_generation, train=False)
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda"))
        else:
            logger.warning(f"No model was loaded as mode={mode}")
            model = None
        self.model = model
        self.model_loaded = True

    """
    Training of model
    """
    def train(self, config: DictConfig):
        # load model
        if config.query_generation.causal:
            model = CausalModel(causal_config=config.query_generation, train=True)
        else:
            model = Seq2SeqModel(seq2seq_config=config.query_generation, train=True)
        
        # create datasets
        train_set = self.create_dataset_for_split(model.tokenizer, config, "train")
        dev_set = self.create_dataset_for_split(model.tokenizer, config, "dev")

        # train
        model.train(train_set, dev_set)
        model.save()

    def create_dataset_for_split(self, tokenizer: PreTrainedTokenizer, config: DictConfig, split: str) -> DatasetSeq2Seq | DatasetCausalModel:
        # load data
        data_input_path = config.query_generation.data_path[split]
        data = load_jsonl(data_input_path)
        inputs = [it["input"] for it in data]
        outputs = [it["output"] for it in data]

        # encode inputs and outputs
        if config.query_generation.causal:
            return self._create_dataset_causal(
                inputs=inputs,
                outputs=outputs,
                tokenizer=tokenizer,
                config=config,
                split=split
            )
        else:
            return self._create_dataset_seq2seq(
                inputs=inputs,
                outputs=outputs,
                tokenizer=tokenizer,
                config=config,
                split=split
            )
            
    def _create_dataset_causal(self, inputs: List[str], outputs: List[str], tokenizer: PreTrainedTokenizer, config: DictConfig, split: str) -> DatasetSeq2Seq:
        inputs = [self.represent_input_causal(question=i, tokenizer=tokenizer) for i in inputs]
        input_encodings, labels = DatasetCausalModel.prepare_encodings_train(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_length=config.query_generation.max_length,
        )
        dataset_length = len(inputs)
        dataset = DatasetCausalModel(input_encodings, labels, dataset_length)
        logger.info(f"Derived {split} set with {len(dataset)} instances.")
        return dataset
    
    def represent_input_causal(self, question: str, tokenizer: AutoTokenizer) -> str:
        prompt = self.prompt.render(
            question=question,
            sql_schema=self.sql_schema
        )
        dialog = [{
            "role": "system",
            "content": self.instruction
        },{
            "role": "user",
            "content": prompt
        }]

        # apply chat template
        input_str = tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=False,
        )
        return input_str

    @staticmethod
    def _create_dataset_seq2seq(inputs: List[str], outputs: List[str], tokenizer: PreTrainedTokenizer, config: DictConfig, split: str) -> DatasetSeq2Seq:
        input_encodings, output_encodings = DatasetSeq2Seq.tokenize(
            tokenizer=tokenizer,
            inputs=inputs,
            outputs=outputs,
            max_input_length=config.query_generation.max_input_length,
            max_output_length=config.query_generation.max_output_length,
        )
        dataset_length = len(inputs)
        dataset = DatasetSeq2Seq(input_encodings, output_encodings, dataset_length)
        logger.info(f"Derived {split} set with {len(dataset)} instances.")
        return dataset

    @staticmethod
    def derive_data(config: DictConfig):
        def derive_data_for_split(config: DictConfig, split: str):
            benchmark_dir = config.benchmark.benchmark_dir
            persona_dir = os.path.join(benchmark_dir, split)
            personas = get_persona_names(persona_dir)

            question_sql_pairs = set()
            for persona in tqdm(personas):
                # init paths and query execution
                obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
                questions_path = f"{persona_dir}/{persona}/questions.json"
                query_execution = QueryExecution(obs_events_csv_path=obs_events_csv_path)
                data = load_json(questions_path)

                # run all queries
                for instance in data:
                    # derive data from instance 
                    sql_query = instance["sql_query"]
                    reference_date = instance["reference_date"]
                    question = instance["question"]
                    
                    # eval query
                    sql_query = query_execution.adjust_sql_query_reference_data(sql_query, reference_date)
                    try:
                        res_df = query_execution.db.query(sql_query).df()
                        answer = query_execution.parse_query_result(res_df, json_serializable=True)
                    except Exception as e:
                        answer = None
                        logger.warning(f"Error with sql_query={sql_query}: {e}")
                        instance["failed"] = True
                        instance["error"] = str(e)
                        continue
                    relaxed_hit_at_1 = hit_at_1(answer, instance["answers"], relax_factor=0.2)
                    if relaxed_hit_at_1:
                        question_sql_pairs.add((question, sql_query))

            # derive training data
            data = [
                {"input": question, "output": sql_query}
                for question, sql_query in question_sql_pairs
            ]
            logger.debug(f"Derived {len(data)} instances.")
            data_output_path = config.query_generation.data_path[split]
            store_jsonl(data_output_path, data)
            logger.debug(f"Stored data for {split}.")

        derive_data_for_split(config, "train")
        derive_data_for_split(config, "dev")


def main():
    # check if provided options are valid
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python query_generation.py <FUNCTION> [<CONFIG>]"
        )
    config_path = "config/perqa/query_generation_openai.yml" if len(sys.argv) < 3 else sys.argv[2]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # run
    function = sys.argv[1]
    if function.startswith("--dev"):
        qg = QueryGeneration(config)
        qg.run(split="dev")
    elif function.startswith("--test"):
        qg = QueryGeneration(config)
        qg.run(split="test")
    elif function.startswith("--derive_data"):
        QueryGeneration.derive_data(config)
    elif function.startswith("--create_datasets"):
        QueryGeneration.create_datasets(config)
    elif function.startswith("--train"):
        qg = QueryGeneration(config)
        qg.train(config)
    else:
        raise Exception(f"Unknown function {function}.")


if __name__ == "__main__":
    main()