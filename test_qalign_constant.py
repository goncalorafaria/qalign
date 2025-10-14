from qalign.model import RemoteVLLM
from qalign.reward import ConstantReward
from qalign.base import QAlign
# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://g3045.hyak.local:8001",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    max_prompt_length=100,
    max_new_tokens=50,
)

reward = ConstantReward(1.0)

chain = QAlign(
    model=model,
    reward=reward,
    beta=1.0, 
)

t = [{"role": "user", "content": "What district is Guimarães in? Answer by explaining all of portuguese history in 1000 words."}]

results =chain.run(
    conversations=[t]*2,
    steps=1,
    use_tqdm=True,
)

#print(results)