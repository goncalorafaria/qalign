from typing import List, Callable
import numpy as np
from typing import List, Optional, Dict
import requests
import aiohttp
import asyncio
from qalign.utils.list import chunked
from itertools import islice
import os
from typing import List
import gc
import math

from scipy.special import logit
#from quest.utils.logger import fix_loggers
from qalign.utils.data import get_loader
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
#fix_loggers(name="transformers")

import torch
from typing import Dict
from torch import nn


DEBUG = os.getenv("DEBUG", "False").lower() == "true"
class Reward:
    """
    The base class for reward evaluation.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name.replace("/", "-").split(".")[0]

    def evaluate(
        self,
       conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.

        """
        raise NotImplementedError


class ConstantReward(Reward):
    """
    A class for a constant reward.

    Attributes:
        reward (float): The reward value.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, reward: float):
        """
        The constructor for ConstantReward class.

        Args:
            reward (float): The reward value.

        """
        self.reward = reward
        super().__init__(f"constant:{self.reward}")

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """
 

        return [self.reward for _ in range(len(conversations))]

    def set_context(self, *args, **kwargs):
        pass


class BackwardReward(Reward):
    """
    A class for a reward based on a backward model.

    Attributes:
        model (Model): The backward model to use for reward evaluation.

    Methods:
        evaluate: Evaluates the reward for a list of candidates.

    """

    def __init__(self, model: Reward):
        """
        The constructor for BackwardReward class.

        Args:
            model (Model): The backward model to use for reward evaluation.

        """
        self.model = model
        super().__init__(f"b:{self.model.get_name()}")

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates the reward for a list of candidates.

        Args:
            candidates (List[str]): A list of candidate strings.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward values for each candidate.

        """

        return [-s for s in self.model.evaluate(conversations, **kwargs)]


class RewardMix(Reward):
    def __init__(
        self,
        rewards: List[Reward],
        mixing_weights: List[float] = None,
    ):
        self.rewards = rewards
        self.mixing_weights = (
            mixing_weights if mixing_weights is not None else [1.0] * len(rewards)
        )

        assert len(self.rewards) == len(
            self.mixing_weights
        ), "The number of rewards must match the number of mixing weights."

        super().__init__(f"mix:{','.join([r.get_name() for r in rewards])}")

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:

        ## TODO THIS SHOULD BE DONE IN PARALLEL !!!!!!!
        evaluations = [r.evaluate(conversations, **kwargs) for r in self.rewards]

        return (
            np.stack(
                [
                    np.array(e) * w
                    for e, w in zip(
                        evaluations,
                        self.mixing_weights,
                    )
                ],
                axis=0,
            )
            .sum(0)
            .tolist()
        )


class RewardStatic(Reward):
    def __init__(
        self,
        reward_func: Callable,
    ):
        self.reward_func = reward_func

        super().__init__(f"static:{reward_func.__name__}")

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:

        ## TODO THIS SHOULD BE DONE IN PARALLEL !!!!!!!
        evaluations = self.reward_func(conversations, **kwargs)

        return evaluations


class RewardSum(RewardMix):
    def __init__(
        self,
        reward_funcs: List[Callable],
    ):
        self.rewards = [RewardStatic(f) for f in reward_funcs]

        super().__init__(rewards=self.rewards)


"""
    output= requests.post("http://g3103.hyak.local:8000/classify",
        json={
            "input": List[str],
        },
    )
"""
class RemoteReward(Reward):
    def __init__(
        self,
        server_url: str,
        model_path: str,
        max_retries: int = 50,
        polling_interval: float = 0.5,
        timeout: float = 300,  # 5 minutes default timeout
        #batch_size=64, 
        server_format: str = None,  # Auto-detect if None
        max_concurrent_requests: int = 256,
    ):
        """
        Client for interacting with the Reward Model Server.

        Args:
            server_url: Base URL of the reward model server (gateway URL)
            model_path: Path/name of the model
            max_retries: Maximum number of status check retries
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait for result in seconds
            server_format: Backend format ("vllm", "sglang", "legacy"). Auto-detected if None.
            max_concurrent_requests: Maximum concurrent requests
        """

        super().__init__(f"rm:{model_path}")
        
        self.max_concurrent_requests = max_concurrent_requests
        self.server_url = server_url.rstrip("/")
        self.max_retries = max_retries
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.model_path = model_path 
        #self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.routes_config = {
            "legacy":{
                "health": "health",
                "evaluate": "classify",
                "arg":"input",
                "parse":lambda x: x["rewards"],
            },
            "vllm": {
                "health": "v1/models",
                "evaluate": "classify",
                "arg":"input",
                "parse":lambda x:logit( x["data"][0]["probs"][0]),
            },
            "sglang": {
                "health": "v1/models",
                "evaluate": "classify",
                "arg":"text",
                "parse":lambda x: x["embedding"][0],
            },
        }
        
        # Auto-detect backend if not specified
        if server_format is None:
            server_format = self._detect_backend()
            print(f"Auto-detected backend: {server_format}")
        
        self.server_format = server_format
        self.routes = self.routes_config[server_format]
        
        # Test connection and get model info
        self._check_health()
    
    def _detect_backend(self) -> str:
        """
        Auto-detect the backend type by querying the gateway's /v1/models endpoint.
        
        Returns:
            str: Backend type ("vllm", "sglang", or "legacy")
        """
        import requests
        
        try:
            # Query the gateway's v1/models endpoint
            response = requests.get(f"{self.server_url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Look for our model in the response
            if "data" in data:
                for model_entry in data["data"]:
                    if model_entry.get("id") == self.model_path:
                        # Check metadata for backend info
                        metadata_list = model_entry.get("metadata", [])
                        if metadata_list and len(metadata_list) > 0:
                            # Get the first server's metadata
                            server_metadata = metadata_list[0].get("metadata", {})
                            backend = server_metadata.get("backend")
                            
                            if backend in ["vllm", "sglang"]:
                                return backend
                            
            # Fallback: Try to detect by testing endpoints
            print("Backend not found in metadata, attempting to detect by testing endpoints...")
            
            # Test for SGLang (has /health endpoint)
            try:
                health_response = requests.get(f"{self.server_url}/health", timeout=5)
                if health_response.status_code == 200:
                    return "sglang"
            except:
                pass
            
            # Default to vllm
            print("Defaulting to vllm format")
            return "vllm"
            
        except Exception as e:
            print(f"Warning: Could not auto-detect backend ({e}), defaulting to vllm")
            return "vllm"

    def _check_health(self):
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/{self.routes['health']}", timeout=self.timeout)
            response.raise_for_status()
            
            if DEBUG:
                print(f"Server[{self.model_path}]: {self.server_url} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")


    def _evaluate(self, payload,use_tqdm=False) -> List[float]:
        """Synchronous wrapper for async evaluation with retries."""
        return asyncio.run(self._evaluate_async(payload,use_tqdm=use_tqdm))

    async def _evaluate_async(self, payload, use_tqdm=False) -> List[float]:
        """
        Submit texts for evaluation with retry logic.

        Args:
            payload: List of payloads to evaluate

        Returns:
            List of reward scores

        Raises:
            RuntimeError: If all retry attempts fail
            TimeoutError: If evaluation times out
        """
        from tqdm.asyncio import tqdm as tqdm_asyncio

        # Create ONE session with connection pooling for ALL requests
        # Since gateway is a single host, set limits based on max_concurrent_requests
        # Connection reuse means actual connections needed < max_concurrent_requests
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,  # Total connection pool size
            limit_per_host=self.max_concurrent_requests,  # Max connections to gateway
            ttl_dns_cache=600,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            async def _make_request_with_retries(p):
                for attempt in range(self.max_retries):
                    try:
                        async with session.post(
                            f"{self.server_url}/{self.routes['evaluate']}",
                            json=p,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            return data
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                        await asyncio.sleep(self.polling_interval)

            # Limit concurrent requests using a semaphore (no batching for-loop)
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_request(p):
                async with semaphore:
                    return await _make_request_with_retries(p)

            tasks = [limited_request(p) for p in payload]

            if use_tqdm:
                # Use tqdm with gather to preserve order
                from tqdm import tqdm
                
                # Create a wrapper to update progress bar
                completed_count = [0]
                pbar = tqdm(total=len(tasks), desc="Evaluating")
                
                async def tracked_task(task):
                    result = await task
                    completed_count[0] += 1
                    pbar.update(1)
                    return result
                
                results = await asyncio.gather(*[tracked_task(task) for task in tasks])
                pbar.close()
            else:
                results = await asyncio.gather(*tasks)

            if DEBUG:
                print(results[0])

            # Extract rewards from results
            rewards = [self.routes["parse"](result) for result in results]

            return rewards

    def evaluate(self, conversations, use_tqdm=False, **kwargs):
        """
        Evaluate candidates using the reward model.

        Args:
            candidates: List of candidate texts to evaluate
            use_tqdm: Whether to use progress bar (not implemented yet)
            **kwargs: Additional keyword arguments

        Returns:
            List of reward scores
        """
       
        
        # Use the configured batch size directly

        # Create payloads with texts and context
        payloads = [
            {"input": 
                self.tokenizer.apply_chat_template(
                t,
                tokenize=False,
                add_generation_prompt=False,
                )
          } for t in conversations
        ]
        
        if DEBUG:
            print("<rm> texts:",repr(payloads[0])) 
            
        # Pack payloads into batches
        packed_payload = [
            {
                self.routes["arg"]: p["input"],
                "model": self.model_path
            }
            for p in payloads
        ]
        
        

        if DEBUG:   
            print("<rm> payload:",packed_payload[0])
        # Evaluate using the async method (wrapped synchronously)
        results = self._evaluate(packed_payload,use_tqdm=use_tqdm)

        return results


class RewardModel(Reward):

    # applies the model only on outputs r(y)
    """
    RewardModel class represents a reward model based on a pre-trained Hugging Face model.

    Args:
        model_path (str): The path to the pre-trained model.
        batch_size (int, optional): The batch size for inference. Defaults to 32.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.

    Attributes:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to use for inference.

    Methods:
        evaluate(candidates: List[str]) -> List[float]:
            Evaluates a list of candidate sequences and returns a list of reward values.

    """

    def __init__(
        self,
        model_path: str,
        device: int = 0,
        task: str = "text-classification",
        clamp: float = 40,
        dtype=torch.bfloat16,
        use_flash_attention: bool = True,
        device_count=1,
        max_length: int = 1024,
        batch_size: int = 32,
    ):

        super().__init__(f"rm:{model_path}")

        self.batch_size = batch_size
        self.device = device
        """self.model = pipeline(
            task,
            model=model_path,
            device=device,
        )"""
        self.kwargs = {"batch_size": self.batch_size}
        self.clamp = clamp
        self.max_length = max_length

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.bos_token_id
            )  # THIS IS ACTUALLY REALLY IMPORTANT :) THIS HIDDEN NIGHTMARE DONT USE EOS. - w/ AR models in batch we may have padding in the beginig
            self.tokenizer.pad_token = self.tokenizer.bos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=dtype,
            pad_token_id=self.tokenizer.pad_token_id,
            use_flash_attention_2=use_flash_attention,
            # device_map="auto",
        )

        if device_count > 1:

            self.model = nn.DataParallel(
                self.model, device_ids=(np.arange(device_count) + self.device).tolist()
            )

        self.device = torch.device(
            f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        candidates: List[str],
        use_tqdm=False,
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.
            accepted_indices (List[int]): The list of indices of accepted candidates.
            batch_size (int, optional): The batch size for inference. Defaults to 32.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """

        loader = get_loader(
            candidates,
            self.tokenizer,
            use_tqdm=use_tqdm,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        rewards = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = np.clip(
                    outputs.logits[:, 0].float().cpu().numpy(),
                    -self.clamp,
                    self.clamp,
                ).tolist()

                del outputs, input_ids, attention_mask
                gc.collect()
                torch.cuda.empty_cache()

                rewards.extend(logits)

        return rewards


class ClassificationRewardModel(RewardModel):

    # applies the model only on outputs r(y)
    """
    RewardModel class represents a reward model based on a pre-trained Hugging Face model.

    Args:
        model_path (str): The path to the pre-trained model.
        batch_size (int, optional): The batch size for inference. Defaults to 32.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.

    Attributes:
        model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to use for inference.

    Methods:
        evaluate(candidates: List[str]) -> List[float]:
            Evaluates a list of candidate sequences and returns a list of reward values.

    """

    def __init__(self, **rm_kwargs):
        super().__init__(**rm_kwargs)
        self.name = "c" + self.name

    def evaluate(
        self,
        conversations: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """
        Evaluates a list of candidate sequences and returns a list of reward values.

        Args:
            candidates (List[str]): The list of candidate sequences to evaluate.

        Returns:
            List[float]: The list of reward values for each candidate sequence.

        """
        texts = [ self.tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        ) for conv in conversations]

        return super().evaluate(texts, **kwargs)
