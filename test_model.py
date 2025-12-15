from qalign.model import RemoteVLLM


# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://g3098.hyak.local:8080",
    model_path="Qwen/Qwen2.5-Math-1.5B-Instruct",
)

print("starting endpoint")

t= [{"role": "user", "content": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"}]


print(t)

print(model.ancestral([t]))

