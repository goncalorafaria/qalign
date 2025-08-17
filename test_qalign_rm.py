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

model = RemoteVLLM(
    server_url="http://g3090.hyak.local:8080",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    max_prompt_length=1000,
    max_new_tokens=1000,
)

reward = RemoteReward(
    server_url="http://g3090.hyak.local:8080",
    model_path='/gscratch/ark/graf/quest-rlhf/qflow/rm/artifacts/llama3/8b8b/gsm8k/full/reward/',
    max_retries=50,
    polling_interval=0.5,
    timeout=300,
    batch_size=8,
)

chain = QAlign(
    model=model,
    reward=reward,
    beta=1.0, 
)

question = "Joana has 10 apples. She gives it to The Lord of Fire which multiplies them by 2 every 10 seconds. One in five of the apples are poisoned and will kill anyone who eats them. All of the apples will be eaten by a hungry crowd. How many people die after 50 seconds?"
steps = 8

t = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,
)

callback = AsyncAnimateLLMResponsesCallback(
    prompt=question,
    total_steps=steps,
)

with callback:
    results =chain.run(
        prompts=[t],
        steps=steps,
        callbacks=[callback],
    )

# 
state_path = [ x["text"] for x in results.state_path[0] if x["accept"]] 




#animate_llm_responses(state_path,prompt=question)



import pdb; pdb.set_trace()