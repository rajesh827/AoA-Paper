"""
Username generation strategies.

Four generation methods are implemented:

  1. random_suffix_generate   -- append a random 2-4 digit number, retry
                                  on collision (the naive baseline).
  2. snowflake_generate       -- derive a short, high-entropy alphanumeric
                                  suffix from a Snowflake-style
                                  (timestamp | node_id | sequence) layout,
                                  following the stateless, network-derived
                                  worker-id design described in the
                                  Stateless Snowflake paper.
  3. trie_suggest_generate    -- use the trie's prefix-count to guess a
                                  suffix number that is likely already
                                  past the range of existing collisions,
                                  rather than guessing blindly.
  4. agentic_style_generate   -- a rule-based stand-in for the
                                  LLM-backed "Nominalist" pipeline: applies
                                  several human-oriented transformation
                                  rules (dot notation, underscore, tag
                                  prefix/suffix, year append) and scores
                                  each candidate with a readability
                                  heuristic, returning the best available
                                  one. NOTE: this substitutes a rule
                                  ensemble for an actual LLM call, since no
                                  LLM API is available in this environment;
                                  this substitution is documented in the
                                  paper's Methodology / limitations.

A shared readability_score() heuristic is used both by the agentic-style
generator internally and by the benchmark's readability-axis evaluation.
"""

import random
import time
import re

NICKNAME_TAGS = ["official", "real", "the", "its", "pro", "dev", "art", "tv"]


# ---------------------------------------------------------------------------
# Readability heuristic
# ---------------------------------------------------------------------------

_VOWELS = set("aeiou")

def readability_score(name):
    """
    Heuristic readability score in [0, 1], combining:
      - length closeness to the 6-15 character 'preferred' range reported
        for username systems,
      - digit-to-length ratio (heavy digit padding reads as less human),
      - a crude pronounceability proxy: presence of at least one vowel
        every few characters, penalizing long consonant/digit runs.
    This is an automated proxy only, meant to be reported alongside (not
    as a replacement for) human ratings collected via readability_sample.csv.
    """
    if not name:
        return 0.0
    length = len(name)
    length_score = 1.0 - min(abs(length - 10) / 10.0, 1.0)

    digit_count = sum(ch.isdigit() for ch in name)
    digit_ratio = digit_count / length
    digit_score = 1.0 - min(digit_ratio * 1.5, 1.0)

    # longest run of consonants/digits without a vowel break
    longest_run = 0
    current_run = 0
    for ch in name:
        if ch in _VOWELS:
            current_run = 0
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)
    pronounce_score = 1.0 - min(longest_run / 8.0, 1.0)

    return round(0.4 * length_score + 0.3 * digit_score + 0.3 * pronounce_score, 4)


# ---------------------------------------------------------------------------
# 1. Random suffix generation (baseline)
# ---------------------------------------------------------------------------

def random_suffix_generate(base_name, is_taken_fn, max_attempts=50):
    attempts = 0
    while attempts < max_attempts:
        suffix = random.randint(1, 9999)
        candidate = f"{base_name}{suffix}"
        attempts += 1
        if not is_taken_fn(candidate):
            return candidate, attempts
    return None, attempts


# ---------------------------------------------------------------------------
# 2. Snowflake-style suffix generation
# ---------------------------------------------------------------------------

class SnowflakeSuffixGenerator:
    """
    Minimal stand-in for the stateless Snowflake design: derives a
    per-node identifier from a simulated private-IP-like value (rather
    than central coordination), and packs (timestamp_ms, node_id,
    sequence) into a short base36 string appended to the requested name.
    """

    _EPOCH_MS = 1_700_000_000_000  # arbitrary fixed epoch

    def __init__(self, node_id):
        self.node_id = node_id & 0xFFFF   # 16-bit, mirrors the paper's layout
        self._last_ms = -1
        self._seq = 0

    def next_id(self):
        now_ms = int(time.time() * 1000)
        if now_ms == self._last_ms:
            self._seq = (self._seq + 1) & 0x3F  # 6 sequence bits => 64/ms
            if self._seq == 0:
                while now_ms <= self._last_ms:
                    now_ms = int(time.time() * 1000)
        else:
            self._seq = 0
        self._last_ms = now_ms
        raw = ((now_ms - self._EPOCH_MS) << 22) | (self.node_id << 6) | self._seq
        return raw

    @staticmethod
    def _to_base36(n):
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "0"
        out = []
        while n:
            n, r = divmod(n, 36)
            out.append(digits[r])
        return "".join(reversed(out))

    def generate(self, base_name, is_taken_fn, max_attempts=5):
        attempts = 0
        while attempts < max_attempts:
            suffix = self._to_base36(self.next_id())[-6:]  # keep it short
            candidate = f"{base_name}{suffix}"
            attempts += 1
            if not is_taken_fn(candidate):
                return candidate, attempts
        return None, attempts


# ---------------------------------------------------------------------------
# 3. Trie-based suggestion
# ---------------------------------------------------------------------------

def trie_suggest_generate(base_name, trie, is_taken_fn, max_attempts=50):
    """
    Uses the trie's prefix count as a starting guess for a numeric
    suffix, on the theory that if N usernames already share this
    prefix, appending a number near N is more likely to be free than
    guessing from 1, which is the advantage a prefix-aware structure
    offers over a blind linear/random search.
    """
    start_guess = trie.count_with_prefix(base_name)
    attempts = 0
    n = max(start_guess, 1)
    while attempts < max_attempts:
        candidate = f"{base_name}{n}"
        attempts += 1
        if not is_taken_fn(candidate):
            return candidate, attempts
        n += 1
    return None, attempts


# ---------------------------------------------------------------------------
# 4. Agentic-style generation (rule ensemble, LLM call substituted)
# ---------------------------------------------------------------------------

def agentic_style_generate(base_name, is_taken_fn, max_candidates_checked=12):
    """
    Produces a small set of human-oriented candidate variants using
    fixed transformation rules (mirroring the deterministic rule
    pathway of the Nominalist creator agent), scores each with the
    readability heuristic, and returns the highest-scoring available
    candidate.

    IMPORTANT: the actual Nominalist design also queries a generative
    language model for additional creative variants. No LLM API is
    available in this benchmark environment, so that pathway is
    omitted here; only the deterministic rule pathway is measured, and
    this is reported as a limitation rather than presented as a full
    reproduction of the agentic framework's output quality.
    """
    year = random.choice(range(1990, 2012))
    tag = random.choice(NICKNAME_TAGS)
    candidates = [
        base_name,
        f"{base_name}_{random.randint(1, 999)}",
        f"{base_name}.{tag}",
        f"{base_name}_{year}",
        f"{tag}_{base_name}",
        f"{base_name.capitalize()}",
        f"{base_name}{year % 100}",
        f"the_{base_name}",
        f"{base_name}_official",
        f"{base_name}.{random.randint(1,99)}",
    ][:max_candidates_checked]

    scored = []
    checked = 0
    for cand in candidates:
        checked += 1
        if not is_taken_fn(cand):
            scored.append((readability_score(cand), cand))
    if not scored:
        return None, checked
    scored.sort(reverse=True)
    return scored[0][1], checked
