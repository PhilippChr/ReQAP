import os
import glob
import json
import codecs
import torch
import pathlib
import pandas as pd
import yaml
import yaml_include
import bm25s
from jinja2 import Template
from loguru import logger
from typing import Iterable, List, Dict, Any, Callable
from omegaconf import OmegaConf, DictConfig
from datetime import datetime, date, time, timedelta
from transformers.tokenization_utils_base import BatchEncoding


def load_config(config_path: str) -> DictConfig:
    yaml.add_constructor("!include", yaml_include.Constructor())
    with open(config_path, "r") as fp:
        config = yaml.full_load(fp)
    return OmegaConf.create(config)


def to_json_dict(d: Dict | List):
    """
    Converts the given dict/list d into a dictionary that
    can be stored as JSON object.
    """
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                k = str(k)
            if isinstance(v, (datetime, date, time)):
                new_dict[k] = v.isoformat()
            elif isinstance(v, timedelta):
                new_dict[k] = extract_largest_unit(v)
            else:
                new_dict[k] = to_json_dict(v)
        return new_dict
    elif isinstance(d, list):
        return [to_json_dict(item) for item in d]
    elif hasattr(d, "to_dict"):
        return d.to_dict()
    else:
        return d
    

def set_random_seed(random_seed: int = 7) -> None:
    import numpy as np
    np.random.seed(random_seed)
    import torch
    torch.manual_seed(random_seed)
    import random
    random.seed(random_seed)
    

def extract_largest_unit(td: timedelta):
    """
    Extract the value of the largest non-zero unit from a timedelta.
    Priority: days > hours > minutes > seconds.
    """
    if td.days != 0:
        return td.days
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    if hours != 0:
        return hours
    minutes = total_seconds // 60
    if minutes != 0:
        return minutes
    return total_seconds


def batchify(l: List | BatchEncoding, batch_size: int):
    """
    Method to iterate a list or batch encoding in batches.
    """
    if type(l) == list:
        for i in range(0, len(l), batch_size):
            yield l[i:i + batch_size]
    elif type(l) == BatchEncoding:
        encodings_dict = l.data
        num_samples = len(encodings_dict['input_ids'])
        # split each field ('input_ids', 'attention_mask', etc.) into batches
        for i in range(0, num_samples, batch_size):
            batch_dict = {key: value[i:i + batch_size] for key, value in encodings_dict.items()}
            yield BatchEncoding(batch_dict)
    else:
        raise TypeError(f"Input of invalid type {type(l)}.")


def pairwise(iterable: Iterable) -> Iterable:
    """
    Returns an iterable which provides the input iterable in pairs.
    """
    a = iter(iterable)
    return zip(a, a)


def avg(l: Iterable, skip_none=False) -> float:
    if skip_none:
        l = [i for i in l if i is not None]
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def sum_skip_nones(l: Iterable) -> float | int:
    l = [i for i in l if i is not None]
    return sum(l)


def argmin(l: Iterable) -> List[int]:
    """ Returns the list of indices which provide the (same) smallest value, resolving draws. """
    min_val = min((v for v in l if v is not None), default=None)
    if min_val is None:
        return []
    indices = list()
    for i, v in enumerate(l):
        if v == min_val:
            indices.append(i)
    return indices


def argmax(l: Iterable) -> List[int]:
    """ Returns the list of indices which provide the (same) largest value, resolving draws. """
    max_val = max((v for v in l if v is not None), default=None)
    if max_val is None:
        return []
    indices = list()
    for i, v in enumerate(l):
        if v == max_val:
            indices.append(i)
    return indices


def load_json(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def store_json(json_file_path: str, data: Any, indent=4) -> None:
    json_file_dir = os.path.dirname(json_file_path)
    pathlib.Path(json_file_dir).mkdir(parents=True, exist_ok=True)
    with open(json_file_path, "w") as fp:
        json.dump(data, fp, indent=indent)


def load_jsonl(jsonl_file_path: str) -> Iterable:
    data = list()
    with open(jsonl_file_path, "r") as fp:
        for line in fp:
            instance = json.loads(line)
            data.append(instance)
    return data


def store_jsonl(jsonl_file_path: str, data: Iterable, file_mode: str="w") -> None:
    jsonl_file_dir = os.path.dirname(jsonl_file_path)
    pathlib.Path(jsonl_file_dir).mkdir(parents=True, exist_ok=True)
    data = to_json_dict(data)
    with open(jsonl_file_path, file_mode) as fp:
        for instance in data:
            fp.write(json.dumps(instance))
            fp.write("\n")


def store_jsonl_line(fp, instance: Any) -> None:
    fp.write(json.dumps(instance))
    fp.write("\n")


def store_jsonl_lines(fp, data: Iterable) -> None:
    for instance in data:
        fp.write(json.dumps(instance))
        fp.write("\n")


def clear_file(file_path: str) -> None:
    if os.path.exists(file_path):
        open(file_path, "w").close()


def handle_output_file(output_path: str, data: List[Dict], override: bool) -> List[Dict]:
    if not output_path is None and override:
        clear_file(output_path)
    if not override:
        if os.path.exists(output_path):
            with open(output_path, "r") as fp:
                num_cases_done = sum(1 for _ in fp)
            logger.info(f"Resuming with question {num_cases_done}.")
            data = data[num_cases_done:]
            logger.info(f"Length of remaining data {len(data)}.")
    return data
            

def load_txt(txt_file_path: str) -> str:
    with open(txt_file_path, "r", encoding="utf-8") as fp:
        text = fp.read()
    return text


def normalize_str(input_str: str):
    try:
        while '\\u' in input_str:
            input_str = codecs.decode(input_str, 'unicode_escape')
    except Exception as e:
        return input_str
    input_str = input_str.strip()
    input_str = input_str.lower()
    return input_str


def num_lines(file_path: str) -> int:
    with open(file_path) as fp:
        for i, _ in enumerate(fp):
            pass
    return i + 1


def duration_to_s(duration_str: str):
    if type(duration_str) == str and len(duration_str.split(":")) == 3:
        h, m, s = duration_str.split(":", 2)
        return (int(h) * 60 + int(m)) * 60 + int(s) * 60
    else:
        return duration_str


def flatten_nested_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to flatten a nested pandas dataframe.
    Used to flatten the event_data column of events.
    Flattens only one level.
    Taken from:
    https://stackoverflow.com/questions/39899005/how-to-flatten-a-pandas-dataframe-with-some-columns-as-json
    """

    df = df.reset_index()

    # search for columns to explode/flatten
    s = (df.map(type) == list).all()
    list_columns = s[s].index.tolist()
    
    s = (df.map(type) == dict).all()
    dict_columns = s[s].index.tolist()
    
    new_columns = []
    
    for col in dict_columns:
        # explode dictionaries horizontally, adding new columns
        horiz_exploded = pd.json_normalize(df[col]).add_prefix(f'{col}.')
        horiz_exploded.index = df.index
        df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
        new_columns.extend(horiz_exploded.columns) # inplace
    
    for col in list_columns:
        # explode lists vertically, adding new columns
        df = df.drop(columns=[col]).join(df[col].explode().to_frame())
        # Prevent combinatorial explosion when multiple
        # cols have lists or lists of lists
        df = df.reset_index(drop=True)
        new_columns.append(col)
    return df


def get_persona_names(persona_dir: str) -> List[str]:
    personas_regex = os.path.join(persona_dir, "*_persona_*")
    persona_paths = [
        os.path.basename(p)
        for p in sorted(glob.glob(personas_regex))
    ]
    return persona_paths


def load_persona_dict(persona_dict_path: str):
    persona_dict = load_json(persona_dict_path)
    return persona_dict


RELEVANT_PERSONA_KEYS = ["siblings", "friends", "partners", "kids", "pets"]
def prepare_persona_dict(persona_dict: Dict):
    # derive relevant data for EXTRACT input
    relevant_data = dict()
    for key in RELEVANT_PERSONA_KEYS:
        entities = persona_dict.get(key, None)
        if entities is None:
            continue
        entities = [entity if isinstance(entity, str) else entity["name"] for entity in entities]
        relevant_data[key] = entities
    return json.dumps(relevant_data)


def normalize_tensor(tensor, eps=1e-9):
    """
    Function to normalize the input tensor on the last dimension.
    """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)


def move_model(model, device):
    """
    Function to move the model to the given device.
    """
    if device == torch.device("cuda"):
        model.eval()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)
    model.eval()


def make_dir(directory_path):
    """
    Function to make the provided directory path.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def flatten(data):
    if isinstance(data, list):
        pruned = [flatten(item) for item in data if item != []]
        return pruned[0] if len(pruned) == 1 and isinstance(pruned[0], list) and len(pruned[0]) == 1 else pruned
    return data


def tensor_to_list(tensor):
    """
    Function to convert a torch tensor to a list.
    """
    return tensor.detach().cpu().tolist()


def tokenize_code(code_str: str, stopwords: List[str]=list()) -> List[str]:
    """Function to tokenize code string."""
    import tokenize
    from io import BytesIO
    example_bytes = BytesIO(code_str.encode('utf-8'))
    tokenizer_res = tokenize.tokenize(example_bytes.readline)
    next(tokenizer_res)  # skip encoding token
    tokens_list = list()
    try:
        for token in tokenizer_res:
            if token.type == 3:
                string = token.string
                string = string.replace("\"", "").replace("'", "").strip()
                strings = [s for s in string.split() if s not in stopwords]
                tokens_list += strings
            else:
                string = token.string
                if not string:
                    continue
                tokens_list.append(string)
    except tokenize.TokenError as e:
        logger.error(f"Catched tokenize.TokenError with error {e}")
        pass
    return tokens_list        

def tokenize_str(string, stopwords: List[str]=list()):
    """Function to tokenize string on (word-level)."""
    string = string.replace(",", " ")
    string = string.strip()
    return [word.lower() for word in string.split() if not word in stopwords]


def bm25_scoring(query: str, documents: List[str], tokenize_fct: Callable=tokenize_str, stopwords: List[str]=list(), n: int=None) -> List[Dict]:
    """
    Receives a query and a list of strings as input.
    Returns a list of dictionaries as output, which captures
    the ranking and scores of documents, providing their
    original indices.
    If n is set to None (default), the full ranking is returned.
    """
    # tokenize
    tokenized_query = tokenize_fct(query, stopwords)
    tokenized_documents = [tokenize_fct(d, stopwords) for d in documents]

    # init retriever and index
    retriever = bm25s.BM25()
    retriever.index(tokenized_documents, show_progress=False)

    # retrieve
    if n is None:
        n = len(documents)
    n = min(n, len(documents))
    indices, scores = retriever.retrieve([tokenized_query], k=n, show_progress=False)

    # process result
    result = list()
    for idx, score in zip(indices[0], scores[0]):
        result.append({
            "document": documents[idx],
            "index": idx,
            "score": score
        })
    return result


"""
PROMPTS
"""
def load_prompt_template(prompt_path: str) -> Template:
    try:
        with open(prompt_path, "r", encoding="utf-8") as fp:
            template = fp.read()
        return Template(template)
    except Exception as e:
        logger.error(f"Prompt path: {prompt_path}")
        raise e
