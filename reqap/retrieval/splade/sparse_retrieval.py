"""
Taken from: https://github.com/naver/splade
Code for the actual retrieval, given a sparse index.
"""
import os
import torch
import pickle
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer

from reqap.library.library import move_model
from reqap.retrieval.splade.inverted_index import IndexDictOfArray


class SparseRetrieval:
    """
    Retrieval from sparse index.
    """
    def __init__(self, splade_config, model, collection, dim_voc, splade_index_path=None, sparse_index=None, **kwargs):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(splade_config["splade_tokenizer_type"], clean_up_tokenization_spaces=True)
        self.model_path = splade_config["splade_model_type_or_path"]
        self.splade_config = splade_config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        move_model(self.model, self.device)
        self.collection = collection
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.dim_voc = dim_voc
        assert (splade_index_path is not None and sparse_index is None) or (splade_index_path is None and sparse_index is not None)
        if splade_index_path is not None:
            self.sparse_index = IndexDictOfArray(splade_index_path, dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(splade_index_path, "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = sparse_index["index"]
            self.doc_ids = sparse_index["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.index_doc_ids = defaultdict()
        self.index_doc_values = defaultdict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.index_doc_values[key] = value

    def retrieve(self, query, involve_model=True, top_k=10, threshold=0):
        with torch.no_grad():
            processed_query = self.tokenizer(query, return_tensors="pt")
            input_ids = processed_query["input_ids"].squeeze()
            if torch.cuda.is_available():
                processed_query = processed_query.to(torch.device("cuda"))
            if involve_model:
                query_rep = self.model(q_kwargs=processed_query)["q_rep"].squeeze()  # we assume ONE query per batch here
            else:
                query_rep = self.create_query_rep(input_ids, self.dim_voc)
            query_rep_nonzero = torch.nonzero(query_rep)
            # relevant_query_tokens = [self.tokenizer.decode(tid) for tid in query_rep_nonzero]
            values = query_rep[query_rep_nonzero.squeeze()]
            filtered_indexes, scores, derivations = self.score_float(
                self.index_doc_ids,
                self.index_doc_values,
                query_rep_nonzero.cpu().numpy(),
                values.cpu().numpy().astype(np.float32),
                threshold=threshold,
                size_collection=len(self.sparse_index)
            )
            # threshold set to 0 by default, could be better
            if not top_k == 0:
                filtered_indexes, scores, derivations = self.select_topk(filtered_indexes, scores, derivations, k=top_k)
            else:
                scores = -scores
            query_result = [{
                    "id": id_,
                    "score": score,
                    "derivation": [{"token": self.tokenizer.decode(m[0]), "score": m[1]} for m in d],
                    **self.collection[id_]
                }
                for id_, score, d in zip(filtered_indexes, scores, derivations)
            ]
            query_result = sorted(query_result, key=lambda res: res["score"], reverse=True)
            # compute bag of words representation (useful for gaining insights)
            col = torch.nonzero(query_rep).squeeze().cpu().tolist()
            weights = query_rep[col].squeeze().cpu().tolist()
            d = {k: v for k, v in zip(col, weights)}
            sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
            bow_rep = []
            for k, v in sorted_d.items():
                bow_rep.append((self.reverse_voc[k], round(v, 2)))
            return query_result, bow_rep

    @staticmethod
    def select_topk(filtered_indexes, scores, derivations, k):
        # only run partitioning (/sorting but with partitions) in case there are more than k docs with score > 0
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
            derivations = [derivations[i] for i in sorted_]
        # otherwise, all filtered_indexes (the ones with score > 0) are relevant
        else:
            scores = -scores
        return filtered_indexes, scores, derivations

    @staticmethod
    def score_float(
            inverted_index_ids: defaultdict,
            inverted_index_floats: defaultdict,
            indexes_to_retrieve: np.ndarray,
            query_values: np.ndarray,
            threshold: float,
            size_collection: int
        ):
        """
        inverted_index_ids: dictionary mapping token_id to doc_id's
        inverted_index_floats: dictionary mapping token_id to float (value for doc_id at same index in inverted_index_ids)
        indexes_to_retrieve: non-zero indexes for the query
        """
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        derivations = [[] for _ in range(size_collection)]  # remember the token_ids and scores for each document
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx][0]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in range(len(retrieved_indexes)):
                doc_id = retrieved_indexes[j]
                score = query_float * retrieved_floats[j]
                scores[doc_id] += score
                derivations[doc_id].append((local_idx, score))
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # docs for which score > 0
        filtered_derivations = [derivations[i] for i in filtered_indexes]
        return filtered_indexes, -scores[filtered_indexes], filtered_derivations
    
    @staticmethod
    def create_query_rep(input_ids, dim_voc):
        query_rep = torch.zeros(dim_voc)
        query_rep.scatter_(0, input_ids, 1)
        return query_rep

    @staticmethod
    def bag_of_words(input_ids, dim_voc):
        bag_of_words_tensor = torch.zeros(1, dim_voc)
        bag_of_words_tensor.scatter_(1, input_ids, 1)
        return bag_of_words_tensor

