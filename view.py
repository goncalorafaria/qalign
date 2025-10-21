

#!/usr/bin/env python3
"""
Test script for the refactored RemoteReward class.
This script demonstrates the new async POST functionality using direct server URLs.
"""


from qalign.utils.term import animate_llm_responses, AsyncAnimateLLMResponsesCallback
from qalign.utils.math import get_last_number, get_last_option, get_last_math
# google/gemma-3-1b-it
import time
from expkit.storage import DiskStorage, ZipStorage
from expkit import ExpSetup, ExpSetup


setup = ExpSetup(
    storage=DiskStorage(
        base_dir="/gscratch/ark/graf/quest-rlhf/paper-outputs/",
        mode="r",
        # mode="rw"
    ),
)

extract_func=get_last_math
exp=setup["58ee72f5-e0e4-47a3-8aee-8209a2ab5f86"]

print(exp)
j=0
T=15
for i in exp.instances(lazy_iterable=True):
    question = i["input"]["chat_template_prompt"][0]["content"]
    answer = i["input"]["answer"]
    j+=1
    if j < T:
        continue
    
    callback = AsyncAnimateLLMResponsesCallback(
        prompt=question,# + answer ,
        total_steps=exp.meta["steps"],
    )
    
    options ={}
    with callback:
        for output in i["outputs"]:
            
            if output["accept"]:
                a = extract_func(output["text"])
                options[a] = options.get(a, 0) + 1
                response = output["text"] 
            else:
                options[a] = options.get(a, 0) + 1
                
            
            options_str = "[Options: " + ", ".join([f"{k}: {v}" for k, v in options.items()]) + "]"
            callback.add_extra_text(options_str)
            callback.add_response(response)
            time.sleep(0.1)
          
            
             
