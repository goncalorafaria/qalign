from typing import List, Callable
import numpy as np
from typing import List, Optional
import requests
import aiohttp
import asyncio
from qalign.utils.list import chunked
from itertools import islice
import os

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
        candidates: List[str],
        accepted_indices: List[int],
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
        candidates: List[str],
        
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
 

        return [self.reward for _ in range(len(candidates))]

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
        candidates: List[str],
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

        return [-s for s in self.model.evaluate(candidates, **kwargs)]


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
        candidates: List[str],
        **kwargs,
    ) -> List[float]:

        ## TODO THIS SHOULD BE DONE IN PARALLEL !!!!!!!
        evaluations = [r.evaluate(candidates, **kwargs) for r in self.rewards]

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
        candidates: List[str],
        **kwargs,
    ) -> List[float]:

        ## TODO THIS SHOULD BE DONE IN PARALLEL !!!!!!!
        evaluations = self.reward_func(candidates, **kwargs)

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
        batch_size=64, 
        server_format: str = "legacy",
    ):
        """
        Client for interacting with the Reward Model Server.

        Args:
            server_url: Base URL of the reward model server
            model_path: Path/name of the model
            max_retries: Maximum number of status check retries
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait for result in seconds
            reward_type: Type of reward model (contextual, value, qe, etc.)
            batch_size: Size of batches for processing 
        """

        super().__init__(f"rm:{model_path}")
        

        self.server_url = server_url.rstrip("/")
        self.max_retries = max_retries
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.model_path = model_path 
        self.batch_size = batch_size
        
        self.routes = {
            "legacy":{
                "health": "health",
                "evaluate": "evaluate",
                "arg":"texts",
            },
            "vllm": {
                "health": "v1/models",
                "evaluate": "classify",
                "arg":"input",
            }
        }
        
        self.routes = self.routes[server_format]
        
        # Test connection and get model info
        self._check_health()

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


    def _evaluate(self, payload) -> List[float]:
        """Synchronous wrapper for async evaluation with retries."""
        return asyncio.run(self._evaluate_async(payload))
    
    async def _evaluate_async(self, payload) -> List[float]:
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
        async def _make_request_with_retries(p):
            for attempt in range(self.max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
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
        
        # Create tasks for all requests and run them concurrently
        tasks = [_make_request_with_retries(p) for p in payload]
        results = await asyncio.gather(*tasks)
        
        # Extract rewards from results
        rewards = [r for result in results for r in result["rewards"]]
        return rewards

    def evaluate(self, candidates, use_tqdm=False, **kwargs):
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
        temp_batch = self.batch_size

        # Create payloads with texts and context
        payloads = [
            {"texts": t, "context": ""} for t in candidates
        ]
        
        if DEBUG:
            print("<rm> texts:",repr(payloads[0]["texts"])) 
            
        # Pack payloads into batches
        packed_payload = [
            {
                self.routes["arg"]: [p["texts"] for p in packed],
                "context": [p["context"] for p in packed],
                "model": self.model_path,
            }
            for packed in chunked(payloads, temp_batch)
        ]

        # Evaluate using the async method (wrapped synchronously)
        results = self._evaluate(packed_payload)

        return results
