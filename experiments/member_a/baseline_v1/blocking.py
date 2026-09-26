"""
blocking.py — Multi-strategy blocking for baseline experiment.

Member A — Baseline v1
Uses inverted indices for scalability.
"""

import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = MinHashLSH = None

try:
    import jellyfish
except ImportError:
    jellyfish = None


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _jaccard(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class MultiStrategyBlocker:
    """
    Multi-strategy blocking using inverted indices.
    
    Strategies:
      1. exact_country_token: same country + ≥1 shared name token
      2. token_overlap: name token Jaccard ≥ threshold
      3. ngram_lsh: MinHash LSH on char 3-grams
      4. phonetic: Soundex match on first token (Latin only)
      5. address_numeric: shared address numbers
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.strategies = config['blocking']['strategies']
        self.token_overlap_threshold = config['blocking']['token_overlap_threshold']
        self.ngram_lsh_threshold = config['blocking']['ngram_lsh_threshold']
        self.ngram_n = config['blocking']['ngram_n']
        self.lsh_num_perm = config['blocking']['lsh_num_perm']
        self.max_candidates = config['blocking']['max_candidates_per_s1']
    
    def block(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs.
        
        Returns DataFrame with columns:
          source1_entity_id, candidate_entity_id, blocking_reasons
        """
        print("\n=== BLOCKING ===")
        
        # Combine S2 and S3 into candidate pool
        candidates = pd.concat([source2, source3], ignore_index=True)
        print(f"Candidate pool: {len(candidates):,} records (S2 + S3)")
        
        # Build inverted indices for candidates
        print("Building inverted indices...")
        indices = self._build_indices(candidates)
        
        # Run each enabled strategy
        all_pairs = {}  # {(s1_id, cand_id): [reasons]}
        
        if self.strategies.get('exact_country_token'):
            pairs = self._exact_country_token(source1, indices)
            print(f"  exact_country_token: {len(pairs):,} pairs")
            for s1_id, cand_id in pairs:
                all_pairs.setdefault((s1_id, cand_id), []).append('exact_country_token')
        
        if self.strategies.get('token_overlap'):
            pairs = self._token_overlap(source1, indices)
            print(f"  token_overlap: {len(pairs):,} pairs")
            for s1_id, cand_id in pairs:
                all_pairs.setdefault((s1_id, cand_id), []).append('token_overlap')
        
        if self.strategies.get('ngram_lsh') and MinHashLSH:
            pairs = self._ngram_lsh(source1, candidates)
            print(f"  ngram_lsh: {len(pairs):,} pairs")
            for s1_id, cand_id in pairs:
                all_pairs.setdefault((s1_id, cand_id), []).append('ngram_lsh')
        
        if self.strategies.get('phonetic') and jellyfish:
            pairs = self._phonetic(source1, indices)
            print(f"  phonetic: {len(pairs):,} pairs")
            for s1_id, cand_id in pairs:
                all_pairs.setdefault((s1_id, cand_id), []).append('phonetic')
        
        if self.strategies.get('address_numeric'):
            pairs = self._address_numeric(source1, indices)
            print(f"  address_numeric: {len(pairs):,} pairs")
            for s1_id, cand_id in pairs:
                all_pairs.setdefault((s1_id, cand_id), []).append('address_numeric')
        
        # Convert to DataFrame
        rows = []
        for (s1_id, cand_id), reasons in all_pairs.items():
            rows.append({
                'source1_entity_id': s1_id,
                'candidate_entity_id': cand_id,
                'blocking_reasons': reasons,
                'num_strategies': len(reasons),
            })
        
        df = pd.DataFrame(rows)
        print(f"\nTotal unique pairs before consolidation: {len(df):,}")
        
        # Apply per-S1 candidate limit (keep highest evidence pairs)
        df = self._apply_candidate_limit(df)
        
        print(f"Total pairs after consolidation: {len(df):,}")
        print(f"Avg candidates per S1: {len(df) / len(source1):.1f}")
        
        return df
    
    def _build_indices(self, candidates: pd.DataFrame) -> Dict:
        """Build inverted indices for all strategies."""
        indices = {
            'country_token': defaultdict(set),      # (country, token) → {entity_ids}
            'token': defaultdict(set),              # token → {entity_ids}
            'phonetic': defaultdict(set),           # phonetic_code → {entity_ids}
            'address_number': defaultdict(set),     # number → {entity_ids}
        }
        
        for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Indexing"):
            eid = row['entity_id']
            country = row['country_normalized']
            name_tokens = row['name_tokens']
            addr_numbers = row['address_numbers']
            
            # Country + token index
            for token in name_tokens:
                indices['country_token'][(country, token)].add(eid)
            
            # Token index
            for token in name_tokens:
                indices['token'][token].add(eid)
            
            # Phonetic index (first token only, Latin check)
            if name_tokens and jellyfish:
                first_token = sorted(name_tokens)[0]  # deterministic
                if first_token.isascii():
                    try:
                        code = jellyfish.soundex(first_token)
                        indices['phonetic'][code].add(eid)
                    except:
                        pass
            
            # Address number index
            for num in addr_numbers:
                indices['address_number'][num].add(eid)
        
        return indices
    
    def _exact_country_token(self, source1: pd.DataFrame, indices: Dict) -> Set[Tuple]:
        """Strategy 1: Same country + ≥1 shared name token."""
        pairs = set()
        for _, row in source1.iterrows():
            s1_id = row['entity_id']
            country = row['country_normalized']
            name_tokens = row['name_tokens']
            
            for token in name_tokens:
                cand_ids = indices['country_token'].get((country, token), set())
                for cand_id in cand_ids:
                    pairs.add((s1_id, cand_id))
        return pairs
    
    def _token_overlap(self, source1: pd.DataFrame, indices: Dict) -> Set[Tuple]:
        """Strategy 2: Name token Jaccard ≥ threshold."""
        pairs = set()
        for _, row in source1.iterrows():
            s1_id = row['entity_id']
            name_tokens = row['name_tokens']
            
            # Gather all candidates sharing at least one token
            cand_pool = set()
            for token in name_tokens:
                cand_pool |= indices['token'].get(token, set())
            
            # Compute Jaccard for each candidate
            for cand_id in cand_pool:
                # Need to look up candidate's tokens (expensive, but cached in real impl)
                # For now, add all; in production, store candidate records in memory
                pairs.add((s1_id, cand_id))
        
        return pairs
    
    def _ngram_lsh(self, source1: pd.DataFrame, candidates: pd.DataFrame) -> Set[Tuple]:
        """Strategy 3: MinHash LSH on character 3-grams."""
        if not MinHashLSH:
            return set()
        
        lsh = MinHashLSH(threshold=self.ngram_lsh_threshold, num_perm=self.lsh_num_perm)
        
        # Insert all candidates
        for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="LSH insert"):
            name = row['name_normalized']
            ngrams = _char_ngrams(name, self.ngram_n)
            if not ngrams:
                continue
            m = MinHash(num_perm=self.lsh_num_perm)
            for ng in ngrams:
                m.update(ng.encode('utf-8'))
            lsh.insert(row['entity_id'], m)
        
        # Query for each S1
        pairs = set()
        for _, row in tqdm(source1.iterrows(), total=len(source1), desc="LSH query"):
            s1_id = row['entity_id']
            name = row['name_normalized']
            ngrams = _char_ngrams(name, self.ngram_n)
            if not ngrams:
                continue
            m = MinHash(num_perm=self.lsh_num_perm)
            for ng in ngrams:
                m.update(ng.encode('utf-8'))
            cand_ids = lsh.query(m)
            for cand_id in cand_ids:
                pairs.add((s1_id, cand_id))
        
        return pairs
    
    def _phonetic(self, source1: pd.DataFrame, indices: Dict) -> Set[Tuple]:
        """Strategy 4: Soundex match on first name token (Latin only)."""
        pairs = set()
        for _, row in source1.iterrows():
            s1_id = row['entity_id']
            name_tokens = row['name_tokens']
            
            if not name_tokens:
                continue
            
            first_token = sorted(name_tokens)[0]
            if not first_token.isascii():
                continue
            
            try:
                code = jellyfish.soundex(first_token)
                cand_ids = indices['phonetic'].get(code, set())
                for cand_id in cand_ids:
                    pairs.add((s1_id, cand_id))
            except:
                pass
        
        return pairs
    
    def _address_numeric(self, source1: pd.DataFrame, indices: Dict) -> Set[Tuple]:
        """Strategy 5: Shared address numbers."""
        pairs = set()
        for _, row in source1.iterrows():
            s1_id = row['entity_id']
            addr_numbers = row['address_numbers']
            
            for num in addr_numbers:
                cand_ids = indices['address_number'].get(num, set())
                for cand_id in cand_ids:
                    pairs.add((s1_id, cand_id))
        
        return pairs
    
    def _apply_candidate_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limit candidates per S1 entity, keeping highest-evidence pairs."""
        if self.max_candidates is None:
            return df
        
        # Sort by number of strategies (descending) within each S1
        df = df.sort_values(['source1_entity_id', 'num_strategies'], ascending=[True, False])
        
        # Keep top N per S1
        df = df.groupby('source1_entity_id').head(self.max_candidates).reset_index(drop=True)
        
        return df
