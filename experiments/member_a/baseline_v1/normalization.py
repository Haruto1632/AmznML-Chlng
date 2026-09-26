"""
normalization.py — Conservative text normalization for baseline experiment.

Member A — Baseline v1
"""

import re
import unicodedata
from typing import List, Set, Tuple

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

# Legal suffix expansion map
LEGAL_SUFFIXES = {
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
}


def normalize_text(text: str, transliterate: bool = True) -> str:
    """Basic text normalization: lowercase, NFD decomposition, whitespace."""
    if not text or not isinstance(text, str):
        return ""
    
    # NFD decomposition (separate base chars from accents)
    text = unicodedata.normalize('NFD', text)
    
    # Transliterate non-Latin scripts to ASCII (Devanagari → romanized)
    if transliterate and unidecode:
        text = unidecode(text)
    
    # Lowercase
    text = text.lower()
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_name(name: str, expand_suffixes: bool = True) -> Tuple[str, Set[str]]:
    """
    Clean and normalize a business name.
    
    Returns
    -------
    normalized : str
        Cleaned name string
    tokens : set[str]
        Set of meaningful tokens (punctuation removed)
    """
    if not name:
        return "", set()
    
    # Normalize
    cleaned = normalize_text(name, transliterate=True)
    
    # & → and
    cleaned = cleaned.replace('&', 'and')
    
    # Remove URLs / domains
    cleaned = re.sub(r'www\.\S+', '', cleaned)
    cleaned = re.sub(r'\.\w{2,3}$', '', cleaned)  # .com, .net, etc.
    
    # Remove punctuation for tokenization
    tokens_text = re.sub(r'[^\w\s]', ' ', cleaned)
    tokens_text = re.sub(r'\s+', ' ', tokens_text).strip()
    
    # Tokenize
    tokens = set(tokens_text.split())
    
    # Expand legal suffixes
    if expand_suffixes:
        expanded_tokens = set()
        for token in tokens:
            if token in LEGAL_SUFFIXES:
                expanded_tokens.add(LEGAL_SUFFIXES[token])
            else:
                expanded_tokens.add(token)
        tokens = expanded_tokens
    
    # Remove stop words / noise
    stop_words = {'the', 'a', 'an', 'of', 'for', 'in', 'at', 'and', 'or'}
    tokens = {t for t in tokens if t and t not in stop_words and len(t) > 1}
    
    return cleaned, tokens


def clean_address(address: str) -> Tuple[str, List[str]]:
    """
    Clean and normalize a business address.
    
    Returns
    -------
    normalized : str
        Cleaned address string
    numbers : list[str]
        Extracted numeric tokens (building numbers, ZIP codes, etc.)
    """
    if not address:
        return "", []
    
    # Normalize
    cleaned = normalize_text(address, transliterate=True)
    
    # Extract all numeric sequences (including hyphenated like ZIP+4)
    numbers = re.findall(r'\d+(?:-\d+)?', cleaned)
    
    # Expand common abbreviations (keep French terms as-is)
    abbrevs = {
        ' st ': ' street ',
        ' rd ': ' road ',
        ' ave ': ' avenue ',
        ' blvd ': ' boulevard ',
        ' dr ': ' drive ',
        ' ln ': ' lane ',
        ' ct ': ' court ',
        ' pl ': ' place ',
    }
    for abbrev, full in abbrevs.items():
        cleaned = cleaned.replace(abbrev, full)
    
    return cleaned, numbers
