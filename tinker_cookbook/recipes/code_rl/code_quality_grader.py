# code_quality_grader.py
"""Efficient code quality grading with caching and sampling."""

import asyncio
import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


def normalize_code(code: str) -> str:
    """Normalize code for hashing - removes variable whitespace and comments."""
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove multi-line strings/comments (rough approximation)
    code = re.sub(r'"""[\s\S]*?"""', '""""""', code)
    code = re.sub(r"'''[\s\S]*?'''", "''''''", code)
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def hash_code(code: str) -> str:
    """Create a hash of normalized code."""
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode()).hexdigest()


@dataclass
class GraderStats:
    """Statistics for monitoring grader efficiency."""
    total_requests: int = 0
    cache_hits: int = 0
    sampled_out: int = 0
    api_calls: int = 0

    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def api_call_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.api_calls / self.total_requests

    def summary(self) -> str:
        return (
            f"Grader stats: {self.total_requests} requests, "
            f"{self.api_calls} API calls ({self.api_call_rate:.1%}), "
            f"{self.cache_hits} cache hits ({self.cache_hit_rate:.1%}), "
            f"{self.sampled_out} sampled out"
        )


class CodeQualityGrader:
    """
    Efficient code quality grader with caching and sampling.

    Reduces API token usage through:
    1. Caching - identical/similar code returns cached score
    2. Sampling - only grades a percentage of submissions

    Example:
        grader = CodeQualityGrader(
            grader_fn=grade_code_with_claude,
            sample_rate=0.2,  # Grade 20% of submissions
        )

        score = await grader.grade(code)
    """

    def __init__(
        self,
        grader_fn: Callable[[str], float],
        sample_rate: float = 1.0,
        cache_size: int = 10000,
        default_score: float = 0.0,
        seed: int | None = None,
    ):
        """
        Args:
            grader_fn: Synchronous function that grades code (e.g., grade_code_with_claude)
            sample_rate: Probability of grading (0.0 to 1.0). Default 1.0 = grade everything.
            cache_size: Maximum number of cached scores.
            default_score: Score to return when skipping due to sampling.
            seed: Random seed for reproducible sampling.
        """
        self.grader_fn = grader_fn
        self.sample_rate = sample_rate
        self.cache_size = cache_size
        self.default_score = default_score

        self._cache: dict[str, float] = {}
        self._cache_order: list[str] = []  # For LRU eviction
        self._rng = random.Random(seed)
        self._stats = GraderStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> GraderStats:
        return self._stats

    def _evict_if_needed(self) -> None:
        """Evict oldest cache entries if over capacity."""
        while len(self._cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)

    def _add_to_cache(self, code_hash: str, score: float) -> None:
        """Add score to cache with LRU tracking."""
        if code_hash in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(code_hash)
        self._cache[code_hash] = score
        self._cache_order.append(code_hash)
        self._evict_if_needed()

    async def grade(
        self,
        code: str,
        question: str | None = None,
    ) -> float:
        """
        Grade code quality, with caching and sampling.

        Args:
            code: The code to grade.
            question: Optional problem statement for context-aware grading.

        Returns:
            Code quality score (0.0 to 1.0), or default_score if sampled out.
        """
        async with self._lock:
            self._stats.total_requests += 1

            # Check cache first
            code_hash = hash_code(code)
            if code_hash in self._cache:
                self._stats.cache_hits += 1
                # Update LRU order
                self._cache_order.remove(code_hash)
                self._cache_order.append(code_hash)
                return self._cache[code_hash]

            # Random sampling - skip with probability (1 - sample_rate)
            if self._rng.random() > self.sample_rate:
                self._stats.sampled_out += 1
                return self.default_score

            # Actually call the grader
            self._stats.api_calls += 1

        # Run grader outside lock (it's slow)
        try:
            # Check if grader accepts question parameter
            import inspect
            sig = inspect.signature(self.grader_fn)
            if 'question' in sig.parameters and question is not None:
                score = await asyncio.to_thread(self.grader_fn, code, question)
            else:
                score = await asyncio.to_thread(self.grader_fn, code)
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            score = self.default_score

        # Cache the result
        async with self._lock:
            self._add_to_cache(code_hash, score)

        return score

    def log_stats(self) -> None:
        """Log current statistics."""
        logger.info(self._stats.summary())


# Convenience factory functions

def create_claude_grader(
    sample_rate: float = 0.2,
    cache_size: int = 10000,
    default_score: float = 0.0,
    seed: int | None = None,
) -> CodeQualityGrader:
    """Create a grader using Claude Code CLI."""
    from tinker_cookbook.recipes.code_rl.claude_code_qual import grade_code_with_claude
    return CodeQualityGrader(
        grader_fn=grade_code_with_claude,
        sample_rate=sample_rate,
        cache_size=cache_size,
        default_score=default_score,
        seed=seed,
    )


def create_gemini_grader(
    sample_rate: float = 0.2,
    cache_size: int = 10000,
    default_score: float = 0.0,
    seed: int | None = None,
) -> CodeQualityGrader:
    """Create a grader using Gemini CLI."""
    from tinker_cookbook.recipes.code_rl.gemini_code_qual import grade_code_with_gemini
    return CodeQualityGrader(
        grader_fn=grade_code_with_gemini,
        sample_rate=sample_rate,
        cache_size=cache_size,
        default_score=default_score,
        seed=seed,
    )
