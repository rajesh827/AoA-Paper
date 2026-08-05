"""
Core data structures used by the verification / distribution side of the
benchmark: a Bloom filter, a trie, and a consistent hash ring.

All three are implemented from scratch in pure Python (stdlib only) so the
benchmark has no dependency on third-party packages that may or may not be
installed in a given environment.
"""

import hashlib
import math
import bisect
from array import array


# ---------------------------------------------------------------------------
# Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Standard Bloom filter using k independent hash functions derived from
    two base hashes (double hashing / "less hashing, same performance",
    Kirsch & Mitzenmacher), avoiding the cost of k separate hash function
    calls per element.
    """

    def __init__(self, expected_items, target_fp_rate=0.01):
        self.n = max(expected_items, 1)
        self.p = target_fp_rate
        self.m = self._optimal_m(self.n, self.p)
        self.k = self._optimal_k(self.m, self.n)
        # bit array stored as a Python 'array' of unsigned bytes acting as bits
        self.num_bytes = (self.m + 7) // 8
        self.bits = bytearray(self.num_bytes)
        self.count = 0

    @staticmethod
    def _optimal_m(n, p):
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(int(math.ceil(m)), 8)

    @staticmethod
    def _optimal_k(m, n):
        k = (m / n) * math.log(2)
        return max(int(round(k)), 1)

    def _hashes(self, item):
        b = item.encode("utf-8")
        h1 = int.from_bytes(hashlib.md5(b).digest()[:8], "little")
        h2 = int.from_bytes(hashlib.sha1(b).digest()[:8], "little")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item):
        for idx in self._hashes(item):
            self.bits[idx // 8] |= (1 << (idx % 8))
        self.count += 1

    def __contains__(self, item):
        for idx in self._hashes(item):
            if not (self.bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

    def memory_bytes(self):
        return self.num_bytes


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    """
    Character-level trie supporting exact membership queries and
    prefix-based lookups (used by the trie-based suggestion generator to
    find how many usernames already share a given prefix).
    """

    def __init__(self):
        self.root = TrieNode()
        self.count = 0

    def add(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        self.count += 1

    def __contains__(self, word):
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
        return node.is_end

    def count_with_prefix(self, prefix):
        """Return number of stored words that begin with `prefix`
        (used to guess a numeric suffix that is likely free)."""
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return 0
        return self._count_subtree(node)

    def _count_subtree(self, node):
        total = 1 if node.is_end else 0
        for child in node.children.values():
            total += self._count_subtree(child)
        return total

    def memory_bytes(self):
        """Rough estimate: count nodes * approximate per-node overhead."""
        return self._node_count(self.root) * 120  # ~120 bytes/node in CPython

    def _node_count(self, node):
        total = 1
        for child in node.children.values():
            total += self._node_count(child)
        return total


# ---------------------------------------------------------------------------
# Consistent Hash Ring
# ---------------------------------------------------------------------------

class ConsistentHashRing:
    """
    Consistent hashing with virtual nodes, following Karger et al.'s
    construction: each physical node is replicated onto several points on
    a hash ring, and a key is routed to whichever ring point is closest in
    the clockwise direction.
    """

    def __init__(self, nodes=None, virtual_replicas=100):
        self.virtual_replicas = virtual_replicas
        self.ring = {}          # hash_value -> physical node
        self.sorted_keys = []   # sorted list of hash values
        self.nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    @staticmethod
    def _hash(key):
        return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "little")

    def add_node(self, node):
        self.nodes.add(node)
        for i in range(self.virtual_replicas):
            h = self._hash(f"{node}#{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node):
        self.nodes.discard(node)
        for i in range(self.virtual_replicas):
            h = self._hash(f"{node}#{i}")
            if h in self.ring:
                del self.ring[h]
                idx = bisect.bisect_left(self.sorted_keys, h)
                if idx < len(self.sorted_keys) and self.sorted_keys[idx] == h:
                    self.sorted_keys.pop(idx)

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]


class NaiveModuloRing:
    """Baseline for comparison: classic mod-based sharding with no
    consistency guarantees, used to quantify how much worse remapping
    is without consistent hashing."""

    def __init__(self, nodes):
        self.nodes = list(nodes)

    def get_node(self, key):
        h = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "little")
        return self.nodes[h % len(self.nodes)]
