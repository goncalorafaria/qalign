#!/usr/bin/env python3
"""
Test script for the refactored RemoteReward class.
This script demonstrates the new async POST functionality using direct server URLs.
"""

from qalign.reward import RemoteReward
from qalign.model import RemoteVLLM
from qalign.base import QAlign

model = RemoteVLLM(
    server_url="http://klone-login01.hyak.local:8080",
    model_path="Qwen/Qwen3-4B-Instruct-2507",
    max_prompt_length=4000,
    max_new_tokens=4096,
)

reward = RemoteReward(
    server_url="http://klone-login01.hyak.local:8080",
    model_path='Qwen/Qwen2.5-Math-RM-72B',
    server_format="sglang",
)

chain = QAlign(
    model=model,
    reward=reward,
    beta=1.0, 
)

question = "Joana has 10 apples. She gives it to The Lord of Fire which multiplies them by 2 every 10 seconds. One in five of the apples are poisoned and will kill anyone who eats them. All of the apples will be eaten by a hungry crowd. How many people die after 50 seconds?"
steps = 8

t = [{"role": "user", "content": question}]

results = chain.run(
    conversations=[t],
    steps=steps,
    use_tqdm=True,
)

state_path = [ x["text"] for x in results.state_path[0] if x["accept"]] 



import pdb; pdb.set_trace()
