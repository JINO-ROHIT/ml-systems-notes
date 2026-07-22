
from array import array
import torch
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams


def print_match_result(label, result, expected_len):
    matched_len = len(result.device_indices)
    status = "[MATCH]" if matched_len == expected_len else "[NO MATCH]"
    print(f"{status} {label}: matched {matched_len} tokens (expected {expected_len})")
    return matched_len


def example_1_basic_prefix_matching():
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Prefix Matching")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    tokens_a = array("q", [100, 200, 300, 400, 500])
    cache.insert(InsertParams(
        key=RadixKey(tokens_a),
        value=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    ))
    
    # exact match
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [100, 200, 300, 400, 500]))
    ))
    print_match_result("exact match", result, 5)
    
    # prefix match where request is longer
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [100, 200, 300, 400, 500, 600, 700]))
    ))
    print_match_result("prefix match where request is longer", result, 5)
    
    # partial match with difference in the end
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [100, 200, 300, 400, 999]))
    ))
    print_match_result("partial match with difference in the end", result, 4)
    
    # no match
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [999, 200, 300, 400, 500]))
    ))
    print_match_result("no match", result, 0)


def example_2_substring_not_matched():
    print("\n" + "="*60)
    print("EXAMPLE 2: Substring Matching(pretty hard to get this to work)")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    tokens_a = array("q", [100, 200, 300, 400, 500])
    cache.insert(InsertParams(
        key=RadixKey(tokens_a),
        value=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    ))
    
    # same tokens but with different prefix
    # [100, 200, 300] is in the middle, not at the start
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [10, 20, 30, 100, 200, 300]))
    ))
    print_match_result("same tokens with different prefix", result, 0)
    print("[100, 200, 300] is in cache but not matched")
    
    # substring at different offset
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [50, 60, 100, 200, 300, 400]))
    ))
    print_match_result("substring at offset 2", result, 0)
    print("[100, 200, 300, 400] is in cache but NOT matched")
    
    # reverse order (same content, different structure)
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", [500, 400, 300, 200, 100]))
    ))
    print_match_result("everse order", result, 0)
    print("same tokens but reversed so NOT matched")


def example_3_multiturn_conversation():
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-turn Conversation Pattern")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    # Turn 1: System prompt + user query
    turn1 = array("q", [1, 2, 3, 4, 5, 100, 101, 102])  # [sys, user_query_1]
    cache.insert(InsertParams(
        key=RadixKey(turn1),
        value=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    ))
    print("Inserted Turn 1: [sys_prompt(1-5), query_1(100-102)]")
    
    # Turn 2: System prompt + history + new query
    # Same system prompt, but now it's at offset 0, followed by history
    turn2 = array("q", [1, 2, 3, 4, 5, 200, 201, 202, 100, 101, 102, 300, 301, 302])
    #                                                 [history_1    ] [new_query    ]
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(turn2)
    ))
    matched = print_match_result("Turn 2 match", result, 8)
    
    if matched == 8:
        print("System prompt reused!")
    else:
        print("System prompt NOT fully reused!")
    
    # What we WANT: Match [1,2,3,4,5] (system prompt) even though
    # the request has more tokens after it


def example_4_rag_pattern():
    print("\n" + "="*60)
    print("EXAMPLE 4: RAG Pattern (Retrieved Documents)")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    # System prompt
    sys_prompt = array("q", [1, 2, 3, 4, 5])
    
    # Document A (retrieved first)
    doc_a = array("q", [100, 101, 102, 103, 104])
    
    # Document B (retrieved later, but same as doc_a)
    doc_b = array("q", [100, 101, 102, 103, 104])  # Same content!
    
    # Request 1: sys_prompt + doc_a + query
    request1 = array("q", list(sys_prompt) + list(doc_a) + [200, 201])
    cache.insert(InsertParams(
        key=RadixKey(request1),
        value=torch.tensor(range(len(request1)), dtype=torch.int64)
    ))
    print("Inserted Request 1: [sys(1-5), doc_a(100-104), query(200-201)]")
    
    # Request 2: sys_prompt + doc_b + query
    # Same doc_a content, but now with different query
    request2 = array("q", list(sys_prompt) + list(doc_b) + [300, 301])
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(request2)
    ))
    matched = print_match_result("Request 2 match", result, 10)
    
    if matched == 10:
        print("     System prompt + document reused!")
    else:
        print("     Not fully reused!")
    
    # Request 3: Different system prompt + doc_a + query
    # Now doc_a is at offset 5, not offset 5
    request3 = array("q", [9, 9, 9] + list(doc_a) + [400, 401])
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(request3)
    ))
    matched = print_match_result("Request 3 match (doc at offset 3)", result, 0)
    
    if matched == 0:
        print("     Document NOT reused (different offset)!")
        print("    This is the agentic limitation!")


def example_5_agentic_tool_use():
    """Example 5: Agentic tool use pattern."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Agentic Tool Use Pattern")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    # System prompt
    sys_prompt = array("q", [1, 2, 3, 4, 5])
    
    # Tool schema
    tool_schema = array("q", [50, 51, 52, 53, 54, 55, 56, 57])
    
    # Request 1: sys + tools + user_query
    request1 = array("q", list(sys_prompt) + list(tool_schema) + [100, 101])
    cache.insert(InsertParams(
        key=RadixKey(request1),
        value=torch.tensor(range(len(request1)), dtype=torch.int64)
    ))
    print("Inserted Request 1: [sys(1-5), tools(50-57), query(100-101)]")
    
    # Request 2: Different sys + same tools + different query
    # Tools are now at offset 8, not offset 5
    request2 = array("q", [9, 9, 9, 9, 9, 9, 9, 9] + list(tool_schema) + [200, 201])
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(request2)
    ))
    matched = print_match_result("Request 2 match", result, 0)
    
    if matched == 0:
        print("     Tool schema NOT reused (different offset)!")
        print("    This is a common agentic pattern!")


def example_6_repeated_content_different_positions():
    print("\n" + "="*60)
    print("EXAMPLE 6: Repeated Content at Different Positions")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    # Common content (e.g., a document, code snippet, etc.)
    common_content = array("q", [100, 200, 300, 400, 500])
    
    # Request 1: common_content at position 0
    request1 = array("q", list(common_content) + [1000, 1001])
    cache.insert(InsertParams(
        key=RadixKey(request1),
        value=torch.tensor(range(len(request1)), dtype=torch.int64)
    ))
    print("Inserted Request 1: [common(100-500), extra(1000-1001)]")
    
    # Request 2: common_content at position 5
    request2 = array("q", [10, 20, 30, 40, 50] + list(common_content) + [2000, 2001])
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(request2)
    ))
    matched = print_match_result("Request 2 match (common at offset 5)", result, 0)
    
    if matched == 0:
        print("     Common content NOT reused (offset 5 vs 0)!")
    
    # Request 3: common_content at position 10
    request3 = array("q", list(range(10)) + list(common_content) + [3000, 3001])
    
    result = cache.match_prefix(MatchPrefixParams(
        key=RadixKey(request3)
    ))
    matched = print_match_result("Request 3 match (common at offset 10)", result, 0)
    
    if matched == 0:
        print("     Common content NOT reused (offset 10 vs 0)!")


def example_7_simulate_cache_hit_rate():
    print("\n" + "="*60)
    print("EXAMPLE 7: Simulated Cache Hit Rate in Agentic Workload")
    print("="*60)
    
    cache = RadixCache.create_simulated()
    
    # Simulate 10 requests with same tool schema but different prefixes
    tool_schema = list(range(50, 60))  # [50, 51, ..., 59]
    
    total_tokens = 0
    cached_tokens = 0
    
    for i in range(10):
        # Each request has: unique_prefix + tool_schema + unique_query
        unique_prefix = list(range(i * 10, i * 10 + 5))  # Different each time
        unique_query = list(range(1000 + i * 10, 1000 + i * 10 + 3))
        
        request = array("q", unique_prefix + tool_schema + unique_query)
        
        result = cache.match_prefix(MatchPrefixParams(
            key=RadixKey(request)
        ))
        
        matched = len(result.device_indices)
        total_tokens += len(request)
        cached_tokens += matched
        
        # Insert the request
        cache.insert(InsertParams(
            key=RadixKey(request),
            value=torch.tensor(range(len(request)), dtype=torch.int64)
        ))
        
        print(f"Request {i+1}: matched {matched}/{len(request)} tokens")
    
    hit_rate = cached_tokens / total_tokens * 100
    print(f"\nOverall cache hit rate: {hit_rate:.1f}%")
    print(f"Total tokens: {total_tokens}, Cached tokens: {cached_tokens}")
    
    if hit_rate < 50:
        print(" Low hit rate due to offset differences!")
    else:
        print("  Good hit rate!")


def main():
    print("="*60)
    
    example_1_basic_prefix_matching()
    example_2_substring_not_matched()
    example_3_multiturn_conversation()
    example_4_rag_pattern()
    example_5_agentic_tool_use()
    example_6_repeated_content_different_positions()
    example_7_simulate_cache_hit_rate()

if __name__ == "__main__":
    main()
