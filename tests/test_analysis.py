from prefixcache_lab.analyze import analyze_rows, demo_rows
from prefixcache_lab.cache_sim import PrefixLRUCache, build_logical_blocks, segment_reuse_statistics, simulate_capacities
from prefixcache_lab.segmentation import split_prompt
from prefixcache_lab.tokenization import UnicodeCodepointTokenizer, longest_common_prefix


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert longest_common_prefix([], [1]) == 0


def test_prompt_segmentation():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "a"},
        {"role": "function", "content": "tool output"},
        {"role": "user", "content": "new"},
    ]
    sections = split_prompt(messages, [{"name": "tool"}])
    assert len(sections["system_prompt"]) == 1
    assert len(sections["tool_description"]) == 1
    assert len(sections["conversation_history"]) == 2
    assert len(sections["tool_result"]) == 1
    assert sections["current_query"][0]["content"] == "new"


def test_demo_has_prefix_reuse():
    results = analyze_rows(demo_rows(), UnicodeCodepointTokenizer())
    assert results[0]["shared_prefix_tokens"] == 0
    assert results[1]["shared_prefix_tokens"] > 0
    assert 0 < results[1]["reuse_ratio"] < 1


def test_lru_requires_continuous_prefix():
    tokenizer = UnicodeCodepointTokenizer()
    rows = demo_rows()
    first = build_logical_blocks(rows[0], tokenizer)
    second = build_logical_blocks(rows[1], tokenizer)
    cache = PrefixLRUCache(10000)
    assert cache.access(first)["hit_tokens"] == 0
    metrics = cache.access(second)
    assert metrics["hit_blocks"] == len(first)
    assert metrics["hit_tokens"] > 0


def test_capacity_and_reuse_statistics():
    tokenizer = UnicodeCodepointTokenizer()
    rows = demo_rows()
    capacities = simulate_capacities(rows, tokenizer, [0, 10000])
    assert capacities[0]["reusable_tokens"] == 0
    assert capacities[1]["reusable_tokens"] > 0
    stats = segment_reuse_statistics(rows, tokenizer)
    assert any(item["reuse_count"] > 0 for item in stats)
