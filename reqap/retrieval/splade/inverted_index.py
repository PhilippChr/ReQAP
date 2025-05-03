"""
Taken from: https://github.com/naver/splade
Code for inverted index based on arrays, powered by numba based retrieval.
"""

import array
import json
import os
import pickle
import h5py
import numpy as np
from loguru import logger

from collections import defaultdict
from tqdm.auto import tqdm


class IndexDictOfArray:
    """
    Inverted index for storing the document collection.
    """
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                logger.info(f"Index at {self.filename} already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)], dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)], dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                logger.info("Done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                logger.info(f"Initializing new index at {self.filename}...")
                # array.array is a more efficient variant of list, which stores only specific types (as provided by e.g. "I", "f")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            logger.info(f"Initializing new index at {self.filename}...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """
        Add a batch of documents to the index.
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        return self.n

    def save(self, dim=None):
        logger.info("Converting index to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        logger.info(f"Saving index to disk at {self.filename}")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        logger.info("Saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
