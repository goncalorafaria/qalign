#!/usr/bin/env python3
"""
Test script for the refactored RemoteReward class.
This script demonstrates the new async POST functionality using direct server URLs.
"""

from qalign.reward import RemoteReward
from qalign.model import RemoteVLLM
from qalign.reward import ConstantReward
from qalign.base import QAlign
from qalign.utils.term import animate_llm_responses, AsyncAnimateLLMResponsesCallback
# google/gemma-3-1b-it
import time


reward = RemoteReward(
    server_url="http://localhost:8000",
    model_path='Skywork/Skywork-Reward-Llama-3.1-8B-v0.2',
    server_format="vllm",
)

question = "Joana has 10 apples. She gives it to The Lord of Fire which multiplies them by 2 every 10 seconds. One in five of the apples are poisoned and will kill anyone who eats them. All of the apples will be eaten by a hungry crowd. How many people die after 50 seconds?"
steps = 8

t = [{"role": "user", "content": question}]


result = reward.evaluate([t])

print(result)
# 0.003173828125

import pdb; pdb.set_trace()
