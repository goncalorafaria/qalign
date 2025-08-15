from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
import requests
import time
 
 
def unflatten_list(flat_data, counts):

    unflattened_translations = []
    start = 0
    for count in counts:
        end = start + count
        unflattened_translations.append(flat_data[start:end])
        start = end

    return unflattened_translations


DEFAULT_TEMPLATE = PromptTemplate.from_template("{prompt}")


class RemoteVLLM:
    def __init__(
        self,
        server_url: str,
        prompt_template: PromptTemplate = DEFAULT_TEMPLATE,
        max_new_tokens: int = 600,
        max_prompt_length: int = 300,
        stop_tokens: list = None,
        temperature: float = 1.0,
        skip_special_tokens: bool = False,
        timeout: float = 300,
        max_retries: int = 5,
    ):
        
        self.server_url = server_url.rstrip("/")
       
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length
        
        self.skip_special_tokens = skip_special_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self._check_health()
        self.model_path =self.check_model_name()
        self.skip_special_tokens=skip_special_tokens
        


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
        )
        
        if stop_tokens is None:
            self.stop_tokens = [self.tokenizer.eos_token]
        else:
            self.stop_tokens = stop_tokens + [self.tokenizer.eos_token]
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token

        
    def check_model_name(self):
        results = self._get("/v1/models")
        
        models={
            'gemma3:1b': 'google/gemma-3-1b-it',
            "qwen3:8b": "Qwen/Qwen3-8B",
        }
    
        self.model_in_use = [ x["id"] for x in results["data"] if x["id"] in models][0]
        
        return  models[self.model_in_use]
        
        
   
        
    def get_prompt(self, **input_data):
        input_data = {
            k: v
            for k, v in input_data.items()
            if k in self.prompt_template.input_variables
        }
        prompt = self.prompt_template.format(**input_data)
        return prompt

    def encode(self, prompt_data):
        prompt_txt = [self.get_prompt(**data) for data in prompt_data]
        tokens = self.tokenize(prompt_txt)
        return tokens

    def tokenize(self, prompt, **tokenizer_kwargs):
        return [
            self.tokenizer.encode(
                p,
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **tokenizer_kwargs
            )
            for p in prompt
        ]

    def effective_detokenize(self, ids):
        prompt_text = self.decode_tokenize(
            ids, skip_special_tokens=False, spaces_between_special_tokens=False
        )
        return prompt_text
    
    def decode_tokenize(self, ids, **kwargs):
        if "skip_special_tokens" not in kwargs:
            kwargs["skip_special_tokens"] = self.skip_special_tokens
        return self.tokenizer.batch_decode(ids, **kwargs)

    def _check_health(self):
        url = f"{self.server_url}/v1/models"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            print(f"Server {self.server_url} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    def _post_with_retries(self, endpoint, payload):
        results = []
        
        for p in payload:
            for attempt in range(self.max_retries):
                try:
                    resp = requests.post(
                        f"{self.server_url}{endpoint}",
                        json=p,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict):
                        data = [data]
                    results.extend(data)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                    time.sleep(1)
        return results

    def _get(self,endpoint):
        request = requests.get(f"{self.server_url}{endpoint}", timeout=self.timeout)
        request.raise_for_status()
        return request.json()
    
    
    def continuation(self, prompt, prefix=None):
        if prefix is None:
            input_data = prompt
        else:
            input_data = [x[0] + x[1] for x in zip(prompt, prefix)]
            ninput = []
            for x in input_data:
                nx = len(x)
                if nx > self.max_prompt_length:
                    ninput.append(x[nx - self.max_prompt_length :])
                else:
                    ninput.append(x)
            input_data = ninput

        #prompt_text = self.decode_tokenize(
        #    input_data, skip_special_tokens=True, spaces_between_special_tokens=False
        #)
        
         ## skip special tokens at False was causing major issues.
        prompt_text = self.effective_detokenize(input_data)
        
       

        payload = [
            {
                "model": self.model_in_use,#self.model_path,
                "prompt": p,
                "temperature": self.temperature,
                "logprobs": 1,
                "max_tokens": self.max_new_tokens,
                "stop": self.stop_tokens,
            }
            for p in prompt_text
        ]

        results = self._post_with_retries("/v1/completions", payload)

        completions = [
            choice["text"] + self.tokenizer.eos_token
            for result in results
            for choice in result.get("choices", [])
        ]

        
        print("prompt:",repr(prompt_text))
        print("completion:",repr(results[0]["choices"][0]["text"]))
  

        completion_ids = [xi[1:] for xi in self.tokenize(completions)]

        return completion_ids 

    def ancestral(
        self,
        input_data,
        n: int = 1,
    ):
        prompts = []
        for data in input_data:
            prompt = self.get_prompt(**data)
            prompts.extend([prompt] * n)

        payload = [
            {
                "model": self.model_in_use,#self.model_path,
                "prompt": p,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature, 
                "n": 1,
            }
            for p in prompts
        ]

        results = self._post_with_retries("/v1/completions", payload)

        completions = [
            choice["text"] for result in results for choice in result.get("choices", [])
        ]
        return unflatten_list(completions, [n] * len(input_data))
