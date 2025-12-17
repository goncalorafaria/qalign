from transformers import AutoTokenizer
import requests
import aiohttp
import asyncio
import time
import os
import logging
import threading
from qalign.utils.list import unflatten_list
from qalign.shared_session import get_shared_session
from qalign.thread_loop_client import ThreadLoopClient

## get env variable DEBUG
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

class RemoteVLLM(ThreadLoopClient):
    def __init__(
        self,
        server_url: str,
        model_path: str,
        max_new_tokens: int = 1024,
        max_prompt_length: int = 1024*3,
        stop_tokens: list = None,
        temperature: float = 1.0,
        timeout: float = 300,
        max_retries: int = 15,
        max_concurrent_requests: int = 256,
    ):
        
        self.server_url = server_url.rstrip("/")
       
    
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length - 1
         
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests


        self.model_path = model_path 
        self._check_health()
        

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
        
        # No instance-level loop: we share a single event loop per worker thread
        # across all clients via ThreadLoopClient.

    def _run_on_loop(self, coro):
        return self._run_on_thread_loop(coro)

    def __getstate__(self):
        """
        Deepcopy/pickle safe: no thread-local runtime objects stored on instance.
        """
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)

    def encode(self, prompt_txt):
        tokens = self.tokenize(prompt_txt)
        return tokens

    def tokenize(self, prompt, **tokenizer_kwargs):
        return [
            self.tokenizer.encode(
                p,
                max_length=self.max_prompt_length+self.max_new_tokens,
                truncation=True,
                add_special_tokens=False,
                **tokenizer_kwargs
            )
            for p in prompt
        ]
    
    def decode_tokenize(self, ids): 
        return self.tokenizer.batch_decode(ids, skip_special_tokens=False, spaces_between_special_tokens=False)

    def _truncate_tokens(self, tokenized_input):
        """
        Truncate tokenized input if it exceeds max_prompt_length.
        Keeps the end of the sequence (most recent tokens).
        
        Args:
            tokenized_input: List of token IDs
            
        Returns:
            Truncated list of token IDs
        """
        if len(tokenized_input) <= self.max_prompt_length:
            return tokenized_input
        
        # Truncate from the beginning, keeping the end
        return tokenized_input[-self.max_prompt_length:]

    def _check_health(self):
        url = f"{self.server_url}/v1/models"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            
            ## check if self.model_path is in the response
            if not len([1 for x in resp.json()["data"] if x["id"] == self.model_path ]):
                raise ConnectionError(f"Model {self.model_path} not found in the response: - {resp.json()} - {url}")
            
            if DEBUG:
                print(f"Server[{self.model_path}]: {self.server_url} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    def _post_with_retries(self, endpoint, payload, use_tqdm=False):
        """Synchronous wrapper for async POST requests with retries."""
        return self._run_on_thread_loop(
            self._post_with_retries_async(endpoint, payload, use_tqdm=use_tqdm)
        )
    
    async def _post_with_retries_async(self, endpoint, payload, use_tqdm=False):
        """
        Submit requests with retry logic and connection pooling.
        
        Args:
            endpoint: API endpoint to call
            payload: List of payloads to send
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of responses
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Get shared session for this (stable) event loop - shared session manager handles caching.
        session = await get_shared_session()
        if session is None or session.closed:
            raise RuntimeError(
                "Shared session not available. "
                "This should not happen - session should auto-initialize."
            )
        
        async def _make_request_with_retries(p):
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.server_url}{endpoint}",
                        json=p,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        if isinstance(data, dict):
                            data = [data]
                        return data
                except Exception as e:
                    # Log connection errors to identify contention issues
                    error_str = str(e).lower()
                    if "connection" in error_str or "timeout" in error_str or "refused" in error_str:
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e} "
                            f"(thread {threading.current_thread().ident})"
                        )
                    elif "session is closed" in error_str or ("closed" in error_str and "session" in error_str):
                        logger.warning(
                            f"Session closed during request, will be recreated on next attempt. "
                            f"Attempt: {attempt + 1}/{self.max_retries}"
                        )
                    else:
                        logger.debug(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                    
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                    await asyncio.sleep(1)
        
        # Limit concurrent requests using a semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def limited_request(p):
            
            async with semaphore:
               
                result = await _make_request_with_retries(p)
              
                
            
                
                return result
        
        tasks = [limited_request(p) for p in payload]
        
        if use_tqdm:
            from tqdm import tqdm
            
            # Create a wrapper to update progress bar
            completed_count = [0]
            pbar = tqdm(total=len(tasks), desc="Processing")
            
            async def tracked_task(task):
                result = await task
                completed_count[0] += 1
                pbar.update(1)
                return result
            
            results = await asyncio.gather(*[tracked_task(task) for task in tasks])
            pbar.close()
        else:
            results = await asyncio.gather(*tasks)
    
        # Flatten results
        flattened_results = []
        for result in results:
            flattened_results.extend(result)
        
        return flattened_results

    async def _post_with_retries_async_original(self, endpoint, payload):
        """Original approach: separate session for each request"""
        async def _make_request_with_retries(p):
            for attempt in range(self.max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.server_url}{endpoint}",
                            json=p,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            if isinstance(data, dict):
                                data = [data]
                            return data
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                    await asyncio.sleep(1)
        
        # Create tasks for all requests and run them concurrently
        tasks = [_make_request_with_retries(p) for p in payload]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened_results = []
        for result in results:
            flattened_results.extend(result)
        
        return flattened_results

    def _get(self, endpoint):
        request = requests.get(f"{self.server_url}{endpoint}", timeout=self.timeout)
        request.raise_for_status()
        return request.json()
    
    
    def continuation(self, prompt, prefix=None, use_tqdm=False):
        if prefix is None:
            input_data = prompt
        else:
            input_data = [x[0] + x[1] for x in zip(prompt, prefix)]
        
        # Truncate all inputs to max_prompt_length
        input_data = [self._truncate_tokens(x) for x in input_data]

        #prompt_text = self.decode_tokenize(
        #    input_data, skip_special_tokens=True, spaces_between_special_tokens=False
        #)
        lengths = [len(x) for x in input_data]
        #print("lengths-prefix:",lengths)
         ## skip special tokens at False was causing major issues.
        prompt_text = self.decode_tokenize(input_data)
        
       

        payload = [
            {
                "model": self.model_path,
                "prompt": p,
                "temperature": self.temperature,
                "logprobs": 1,
                "max_tokens": self.max_new_tokens,
                "stop": self.stop_tokens,
            }
            for p in prompt_text
        ]

        if DEBUG:
            print("model_packet:",payload[0])

        results = self._post_with_retries("/v1/completions", payload, use_tqdm=use_tqdm)

        completions = [
            choice["text"] 
            for result in results
            for choice in result.get("choices", [])
        ]

        completion_ids = [xi for xi in self.tokenize(completions)]
        #print("lengths-completion:",[len(xi) for xi in completion_ids])
        
        return completion_ids 

    def ancestral(
        self,
        input_data,
        n: int = 1,
        use_tqdm=False,
    ):
        prompts =[ self.tokenizer.apply_chat_template(
            chat_template_prompt,
            tokenize=False,
            add_generation_prompt=True,
        ) for chat_template_prompt in input_data]
        
        # Truncate input_data (text) by tokenizing, truncating, then decoding
        tokenized_data = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, max_length=self.max_prompt_length, truncation=True)
            truncated_prompt = self.tokenizer.decode(tokens, skip_special_tokens=False)
            tokenized_data.append(truncated_prompt)
        
        prompts = []
        for prompt in tokenized_data: 
            prompts.extend([prompt] * n)
            
        logger.info(f"prompts count: {len(prompts)}")
        payload = [
            {
                "model": self.model_path,
                "prompt": p,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature, 
                "n": 1,
                "stop": self.stop_tokens,
            }
            for p in prompts
        ]

        results = self._post_with_retries("/v1/completions", payload, use_tqdm=use_tqdm)

        completions = [
            choice["text"] for result in results for choice in result.get("choices", [])
        ]
        return unflatten_list(completions, [n] * len(input_data))

    async def close(self):
        """
        No-op: event loop is managed per thread in ThreadLoopClient.
        Shared sessions are managed per loop by the shared session manager.
        """
        return
    
    def __del__(self):
        """Cleanup on deletion - sessions are managed by shared session manager."""
        return

    def __str__(self):
        return f"RemoteVLLM(model_path={self.model_path}, server_url={self.server_url})"