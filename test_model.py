from qalign.model import RemoteVLLM


# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://g3090.hyak.local:8080",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
)

print(model.ancestral(["What is the capital of France?"]))