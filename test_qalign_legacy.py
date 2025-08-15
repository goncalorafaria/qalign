from quest.model.remote import RemoteVLLM
from quest.reward.base import ConstantReward
from quest.qalign import QAlign
from qalign.utils.experiment import create_vllm_model
# google/gemma-3-1b-it


model = create_vllm_model(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.0,
        max_new_tokens=1000,
        max_prompt_length=1000,
        remote=True,
    )

reward = ConstantReward(0.0)

t = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "What district is Guimarães in?"}],
    tokenize=False,
    add_generation_prompt=True,
)

chain = QAlign(
    input_data=[{"prompt": t}],
    model=model,
    reward=reward,
    beta=1.0, 
)



results =chain.run(
    steps=8,
)

#print(results)