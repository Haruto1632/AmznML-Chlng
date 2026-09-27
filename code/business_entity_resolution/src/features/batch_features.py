"""Batched implementation of baseline_v1's classical string features.

TF-IDF is fitted once on model-training records, saved with the model, and never
refitted on calibration, evaluation or test. Cosines are sparse row products.
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURES = ["name_token_jaccard", "name_exact_normalized", "name_levenshtein",
            "name_jaro_winkler", "name_char3gram_tfidf_cosine", "addr_token_jaccard",
            "addr_numeric_overlap", "addr_char3gram_cosine", "country_exact",
            "num_blocking_strategies", "name_missing", "address_missing"]


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 1.


class BatchFeatures:
    def fit(self, records):
        self.vectorizers = {}
        for column in ("name", "address"):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3),
                                         max_features=5000, lowercase=False, dtype=np.float32)
            corpus = records[column].tolist()
            if not any(len(text) >= 3 for text in corpus):
                raise ValueError(f"Cannot fit {column} TF-IDF: empty training vocabulary")
            vectorizer.fit(corpus)
            self.vectorizers[column] = vectorizer
        return self

    def transform(self, source, pool, pairs):
        if not len(pairs):
            return pd.DataFrame(columns=FEATURES, dtype=np.float32)
        unique, reverse = np.unique(pairs[:, 1], return_inverse=True)
        target = pool.take(unique)
        src_idx = pairs[:, 0]
        a_name = source.name.to_numpy(); b_name = target.name.to_numpy()
        a_addr = source.address.to_numpy(); b_addr = target.address.to_numpy()
        a_tok = [set(t.split()) for t in source.tokens]; b_tok = [set(t.split()) for t in target.tokens]
        a_at = [set(t.split()) for t in source.address]; b_at = [set(t.split()) for t in target.address]
        a_num = [set(t.split()) for t in source.numbers]; b_num = [set(t.split()) for t in target.numbers]
        values = np.empty((len(pairs), len(FEATURES)), dtype=np.float32)
        values[:, 0] = [jaccard(a_tok[a], b_tok[b]) for a, b in zip(src_idx, reverse)]
        values[:, 1] = (a_name[src_idx] == b_name[reverse]) & (a_name[src_idx] != "")
        values[:, 2] = [Levenshtein.normalized_similarity(a_name[a], b_name[b]) if a_name[a] and b_name[b] else 0 for a, b in zip(src_idx, reverse)]
        values[:, 3] = [JaroWinkler.normalized_similarity(a_name[a], b_name[b]) if a_name[a] and b_name[b] else 0 for a, b in zip(src_idx, reverse)]
        for column, feature_col in (("name", 4), ("address", 7)):
            left = self.vectorizers[column].transform(source[column])
            right = self.vectorizers[column].transform(target[column])
            values[:, feature_col] = np.asarray(left[src_idx].multiply(right[reverse]).sum(axis=1)).ravel()
        values[:, 5] = [jaccard(a_at[a], b_at[b]) for a, b in zip(src_idx, reverse)]
        values[:, 6] = [jaccard(a_num[a], b_num[b]) for a, b in zip(src_idx, reverse)]
        ac = source.country.to_numpy()[src_idx]; bc = target.country.to_numpy()[reverse]
        values[:, 8] = (ac == bc) & (ac != "")
        values[:, 9] = [int(x).bit_count() for x in pairs[:, 2]]
        values[:, 10] = (a_name[src_idx] == "") | (b_name[reverse] == "")
        values[:, 11] = (a_addr[src_idx] == "") | (b_addr[reverse] == "")
        return pd.DataFrame(values, columns=FEATURES)
