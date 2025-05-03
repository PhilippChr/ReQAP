"""
Class to create a sparse index of the provided events.
"""
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from loguru import logger

from reqap.library.library import move_model
from reqap.library.csv import initialize_csv_reader
from reqap.retrieval.splade.inverted_index import IndexDictOfArray
from reqap.retrieval.splade.models import Splade


class IndexConstructor:
    """
    Parse personal data and return a list with the events.
    """
    def run(self, splade_persona_config: OmegaConf, model: Splade):
        verbalize_event_data = splade_persona_config.get("splade_verbalize_events", False)
        d_collection = CollectionDataset(data_path=splade_persona_config.obs_events_csv_path, verbalize_event_data=verbalize_event_data)
        d_loader = CollectionDataLoader(
            dataset=d_collection,
            tokenizer_type=splade_persona_config.splade_tokenizer_type,
            max_length=splade_persona_config.splade_max_length,
            batch_size=splade_persona_config.splade_index_batch_size,
            shuffle=False,
            num_workers=10,
            prefetch_factor=4
        )
        indexing = SparseIndexing(model=model, splade_config=splade_persona_config)
        indexing.run(d_loader)


class SparseIndexing:
    """
    Class that processes the entire collection and constructs the corresponding inverted index.
    Based on https://github.com/naver/splade.
    """
    def __init__(self, model, splade_config, dim_voc=None, force_new=True, **kwargs):
        self.model = model
        self.model_path = splade_config["splade_model_type_or_path"]
        self.splade_config = splade_config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        move_model(self.model, self.device)
        self.index_dir = splade_config["splade_index_path"] if splade_config is not None else None
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)

    def run(self, collection_loader, id_dict=None):
        # encode documents
        doc_ids = []
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id", "data"}}
                batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                row = row + count
                batch_ids = batch["id"]
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(), n_docs=len(batch_ids))
        
        # store index or return to caller function
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            logger.info("Done indexing over the corpus...")
            logger.info("Index contains {} posting lists".format(len(self.sparse_index)))
            logger.info("Index contains {} documents".format(len(doc_ids)))
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key], dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            return out


class CollectionDataLoader(DataLoader):
    """
    Dataloader for the collection.
    Based on https://github.com/naver/splade.
    """
    def __init__(self, tokenizer_type, max_length, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, clean_up_tokenization_spaces=True)
        self.max_length = max_length
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, text, data = zip(*[i.values() for i in batch])
        processed_events = self.tokenizer(
            list(text),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True
        )
        return {
            **{k: torch.tensor(v) for k, v in processed_events.items()},
                "id": id_,
                "data": data
        }


class CollectionDataset(Dataset):
    """
    Dataset to iterate over a document/query collection.
    Format per line: format per line: doc_id \t doc.
    The whole collection is loaded into memory.
    """

    def __init__(self, data_path, verbalize_event_data: bool=False):
        logger.info(f"Verbalization of event data set to: {verbalize_event_data}")
        self.verbalize_events = verbalize_event_data
        self.data_path = data_path
        self.id_dict = {}  # dict storing the whole event (with keys `id`, `date` and `text`)
        self.text_dict = {}  # dict storing only the event_data JSON-strings
        self.data_dict = {}  # dict storing the full data of the event
        logger.info(f"Preloading dataset at {data_path}")

        # loading dataset
        with open(data_path, "r") as fp:
            reader = initialize_csv_reader(fp)
            for i, row in enumerate(reader):
                self.id_dict[i] = row["id"]
                if self.verbalize_events:
                    self.text_dict[i] = self.verbalize_event_data(row["event_data"])
                else:
                    self.text_dict[i] = row["event_data"]
                self.data_dict[i] = row
        self.collection_size = len(self.id_dict)

    def __len__(self):
        return self.collection_size

    def __getitem__(self, idx):
        return {
            "id": self.id_dict[idx],
            "event_data": self.text_dict[idx],
            "data": self.data_dict[idx]
        }
    
    def to_df(self):
        """ Converts the dataset into a Pandas DataFrame. """
        df = pd.DataFrame.from_dict(self.data_dict, orient="index")
        df["event_data"] = df["event_data"].apply(lambda x: json.loads(x))
        return df
    
    def verbalize_event_data(self, event_data_str: str) -> str:
        """ Similar to procedure in CE and EXTRACT datasets. """
        event_dict = json.loads(event_data_str)
        event_str = ",\n".join(f"{k}: {json.dumps(v)}" for k, v in event_dict.items())
        event_str = event_str.replace("_", " ")
        return event_str

    class InvalidDataFormat(Exception):
        pass

