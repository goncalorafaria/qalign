import os
from quest.core import Quest
from quest.reward.model import ContextualRewardModel, ValueHead
from quest.qalign import QAlign
from quest.model.vllm import VLLM


# Model configuration
model_path = "allenai/Llama-3.1-Tulu-3-8B-SFT"
model_args = {
    "model_path": model_path,
    "download_dir": os.environ.get("HF_HOME", "/tmp/"),
    "stop_tokens": ["</s>", "<|im_end|>"],
    "temperature": 0.7,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "max_new_tokens": 512,
    "max_prompt_length": 4096,
    "tensor_parallel_size": 1,  # Number of GPUs
    "enable_prefix_caching": True,
    "enforce_eager": True,
}

# Initialize the model
model = VLLM(**model_args)

# Initialize the reward model
reward = ContextualRewardModel(
    model_path="allenai/Llama-3.1-Tulu-3-8B-RM",
    device=1,  ## second gpu
    device_count=1,
)

# Prepare your data
data_batch = [
    {
        "prompt": "<|user|>\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n<|assistant|>\n"
    },
    # Add more examples as needed
]

# Create markov chain
chain = QAlign(
    input_data=data_batch,
    model=model,
    reward=reward,
    beta=1.0,  # Controls exploration vs exploitation
)

# Run
chain_outputs = chain.run(
    steps=10,  # Number of steps
    use_tqdm=True,  # Show progress bar
)

# Print the accepted outputs
print(f"Original prompt: {chain_outputs[0]['input']['prompt']}")
for output in chain_outputs[0]["outputs"]:
    if output["accept"]:
        print(f"Response: {output['text']}")
        print("-" * 50)
