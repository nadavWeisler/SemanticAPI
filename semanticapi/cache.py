"""In-process TTL response cache."""
from __future__ import annotations

from cachetools import TTLCache

from semanticapi.config import CACHE_MAXSIZE, CACHE_TTL

# Shared caches keyed by (endpoint_name, frozenset_of_params)
_similarity_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
_most_similar_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
