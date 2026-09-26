"""
features.py — Feature engineering for baseline experiment.

Member A — Baseline v1
Classical string similarity features.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein, JaroWinkler
except ImportError:
    fuzz = Levenshtein = JaroWinkler = None


def _jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _safe_ratio(a, b):
    """Safe division, returns 0 if denominator is 0."""
    return a / b if b > 0 else 0.0


class FeatureBuilder:
    """Build pairwise features for candidate pairs."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_config = config['features']
        self.tfidf_name = None
        self.tfidf_addr = None
    
    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        source1: pd.DataFrame,
        candidates: pd.DataFrame,  # S2 + S3
    ) -> pd.DataFrame:
        """
        Build feature matrix for all candidate pairs.
        
        Returns DataFrame with columns:
          source1_entity_id, candidate_entity_id, <features>
        """
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create lookup dicts for fast access
        s1_lookup = source1.set_index('entity_id').to_dict('index')
        cand_lookup = candidates.set_index('entity_id').to_dict('index')
        
        # Fit TF-IDF vectorizers on the full corpus
        if self.feature_config.get('name_char3gram_tfidf_cosine'):
            print("Fitting TF-IDF for name char 3-grams...")
            self._fit_tfidf_name(source1, candidates)
        
        if self.feature_config.get('addr_char3gram_cosine'):
            print("Fitting TF-IDF for address char 3-grams...")
            self._fit_tfidf_addr(source1, candidates)
        
        # Compute features for each pair
        print(f"Computing features for {len(candidate_pairs):,} pairs...")
        feature_rows = []
        
        for _, row in tqdm(candidate_pairs.iterrows(), total=len(candidate_pairs), desc="Features"):
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            
            s1_rec = s1_lookup[s1_id]
            cand_rec = cand_lookup[cand_id]
            
            features = self._compute_pair_features(s1_rec, cand_rec, row)
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            feature_rows.append(features)
        
        df = pd.DataFrame(feature_rows)
        
        # Reorder columns (IDs first, then features)
        id_cols = ['source1_entity_id', 'candidate_entity_id']
        feature_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + feature_cols]
        
        print(f"Feature matrix: {df.shape}")
        return df
    
    def _fit_tfidf_name(self, source1: pd.DataFrame, candidates: pd.DataFrame):
        """Fit TF-IDF on name char 3-grams."""
        corpus = list(source1['name_normalized']) + list(candidates['name_normalized'])
        self.tfidf_name = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            max_features=self.config['features']['tfidf_max_features'],
            lowercase=False,  # already lowercased
        )
        self.tfidf_name.fit(corpus)
    
    def _fit_tfidf_addr(self, source1: pd.DataFrame, candidates: pd.DataFrame):
        """Fit TF-IDF on address char 3-grams."""
        corpus = list(source1['address_normalized']) + list(candidates['address_normalized'])
        self.tfidf_addr = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            max_features=self.config['features']['tfidf_max_features'],
            lowercase=False,
        )
        self.tfidf_addr.fit(corpus)
    
    def _compute_pair_features(self, s1_rec: dict, cand_rec: dict, pair_row: dict) -> dict:
        """Compute all features for one pair."""
        features = {}
        
        # Extract fields
        s1_name = s1_rec['name_normalized']
        s1_name_tokens = s1_rec['name_tokens']
        s1_addr = s1_rec['address_normalized']
        s1_addr_numbers = set(s1_rec['address_numbers'])
        s1_country = s1_rec['country_normalized']
        
        cand_name = cand_rec['name_normalized']
        cand_name_tokens = cand_rec['name_tokens']
        cand_addr = cand_rec['address_normalized']
        cand_addr_numbers = set(cand_rec['address_numbers'])
        cand_country = cand_rec['country_normalized']
        
        # --- NAME FEATURES ---
        if self.feature_config.get('name_token_jaccard'):
            features['name_token_jaccard'] = _jaccard(s1_name_tokens, cand_name_tokens)
        
        if self.feature_config.get('name_exact_normalized'):
            features['name_exact_normalized'] = 1.0 if s1_name == cand_name else 0.0
        
        if self.feature_config.get('name_levenshtein') and Levenshtein:
            if s1_name and cand_name:
                max_len = max(len(s1_name), len(cand_name))
                features['name_levenshtein'] = 1.0 - (Levenshtein.distance(s1_name, cand_name) / max_len)
            else:
                features['name_levenshtein'] = 0.0
        
        if self.feature_config.get('name_jaro_winkler') and JaroWinkler:
            if s1_name and cand_name:
                features['name_jaro_winkler'] = JaroWinkler.normalized_similarity(s1_name, cand_name)
            else:
                features['name_jaro_winkler'] = 0.0
        
        if self.feature_config.get('name_char3gram_tfidf_cosine') and self.tfidf_name:
            if s1_name and cand_name:
                vec_s1 = self.tfidf_name.transform([s1_name])
                vec_cand = self.tfidf_name.transform([cand_name])
                cos_sim = cosine_similarity(vec_s1, vec_cand)[0, 0]
                features['name_char3gram_tfidf_cosine'] = cos_sim
            else:
                features['name_char3gram_tfidf_cosine'] = 0.0
        
        # --- ADDRESS FEATURES ---
        if self.feature_config.get('addr_token_jaccard'):
            s1_addr_tokens = set(s1_addr.split()) if s1_addr else set()
            cand_addr_tokens = set(cand_addr.split()) if cand_addr else set()
            features['addr_token_jaccard'] = _jaccard(s1_addr_tokens, cand_addr_tokens)
        
        if self.feature_config.get('addr_numeric_overlap'):
            features['addr_numeric_overlap'] = _jaccard(s1_addr_numbers, cand_addr_numbers)
        
        if self.feature_config.get('addr_char3gram_cosine') and self.tfidf_addr:
            if s1_addr and cand_addr:
                vec_s1 = self.tfidf_addr.transform([s1_addr])
                vec_cand = self.tfidf_addr.transform([cand_addr])
                cos_sim = cosine_similarity(vec_s1, vec_cand)[0, 0]
                features['addr_char3gram_cosine'] = cos_sim
            else:
                features['addr_char3gram_cosine'] = 0.0
        
        # --- COUNTRY FEATURE ---
        if self.feature_config.get('country_exact'):
            features['country_exact'] = 1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
        
        # --- STRUCTURAL FEATURES ---
        if self.feature_config.get('num_blocking_strategies'):
            features['num_blocking_strategies'] = pair_row.get('num_strategies', 0)
        
        return features
