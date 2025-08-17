from qalign.model import RemoteVLLM
from qalign.reward import ConstantReward
from qalign.base import QAlign
# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://g3090.hyak.local:8080",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    max_prompt_length=1000,
    max_new_tokens=1000,
)

reward = ConstantReward(1.0)

chain = QAlign(
    model=model,
    reward=reward,
    beta=1.0, 
)

t = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "What district is Guimarães in?"}],
    tokenize=False,
    add_generation_prompt=True,
)

results =chain.run(
    prompts=[t],
    steps=8,
)



#print(results)