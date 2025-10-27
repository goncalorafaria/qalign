#!/usr/bin/env python3
"""
Test script for RemoteReward truncation functionality.
This script tests that long conversations are truncated to fit within max_prompt_length.
"""

from qalign.reward import RemoteReward

# Create a reward model with a max_prompt_length limit
reward = RemoteReward(
    server_url="http://localhost:8080",
    model_path='graf/q4-223d83d04db5ef64-1-30-8b-m',
    max_prompt_length=128,  # Small limit to test truncation
)

# Create a conversation with a very long response that will exceed the limit
question = "What is the capital of France?"
long_response = "The capital of France is Paris. " * 50  # Make it very long

t = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": long_response}
]

print(f"Original response length: {len(long_response)} characters")
print(f"Max prompt length (tokens): {reward.max_prompt_length}")

# This should not fail even though the response is very long
result = reward.evaluate([t])

print(f"Evaluation successful! Reward: {result}")

# Test with multiple conversations
multiple_conversations = [
    [
        {"role": "user", "content": f"Question {i}"},
        {"role": "assistant", "content": "Response " * (i * 10)}  # Varying lengths
    ]
    for i in range(50)
]

results = reward.evaluate(multiple_conversations)
print(f"Evaluated {len(results)} conversations successfully")
print(f"Results: {results}")

print("\nAll tests passed!")

