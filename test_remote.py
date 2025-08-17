from qalign.model import RemoteVLLM


# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://127.0.0.1:11434",
    
)

print(model.ancestral([{"prompt": "What is the capital of France?"}]))