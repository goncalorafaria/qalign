import pipes
from posix import pipe
from qalign.reward import Reward, RemoteReward
from qalign.model import RemoteVLLM
from typing import List, Dict
import numpy as np
from qalign.base import QAlign
from qalign.utils.math import sample_index, log_prob_index



class RemoteLMReward(Reward):
    
    def __init__(self, server_url: str, model_path: str, full_logprobs: bool = False, **model_kwargs):
        
        model_tag= model_path.replace(".","")
        super().__init__(f"logprobs:{model_tag}")
        self.full_logprobs = full_logprobs
        self.model_client = RemoteVLLM(
            server_url=server_url,
            model_path=model_path,
            track_logprobs=True,
            **model_kwargs,
        )
        
    
    
    def _tokenize_conversations(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        prompts =[ self.model_client.tokenizer.apply_chat_template(
            chat_template_prompt,
            tokenize=True,
            add_generation_prompt=False,
        ) for chat_template_prompt in conversations ]
        
        return prompts
    
    def _tokenize_prompt(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        prompts =[ self.model_client.tokenizer.apply_chat_template(
            chat_template_prompt[:-1],
            tokenize=True,
            add_generation_prompt=True,
        ) for chat_template_prompt in conversations ]
        
        return prompts
        
  
    async def _evaluate_async(self, conversations: List[List[Dict[str, str]]],use_tqdm=False, **kwargs) -> List[float]:
    
    
        all_logprobs = await self.model_client._logprobs_async(conversations, use_tqdm=use_tqdm)
        
        if self.full_logprobs:
            return [ float(np.sum(all_logprobs[i])) for i in range(len(all_logprobs)) ]
        else:
            tokens_prompt = self._tokenize_prompt(conversations)
            tokens_full = self._tokenize_conversations(conversations)
            
            #response_lengths = [ len(all_ids) - len(p_ids) for p_ids, all_ids in zip(tokens_prompt, tokens_full) ]
            prompt_lengths = [ len(p_ids) for p_ids in tokens_prompt ]
            return [ float(np.sum(all_logprobs[i][pi:])) for i, pi in enumerate(prompt_lengths) ]
        
        
    
    def evaluate(self, conversations: List[List[Dict[str, str]]], use_tqdm=False, **kwargs) -> List[float]:
        return self.model_client._run_on_thread_loop(
            self._evaluate_async(conversations, use_tqdm=use_tqdm, **kwargs)
        ) 


class PIReward(Reward):
    
    def __init__(self, ref: RemoteLMReward, reward: RemoteReward, beta: float = 1.0):
        super().__init__(f"pi:{ref.get_name()}:{reward.get_name()}")
        
        self.ref = ref
        self.reward = reward
        self.beta = beta
        
    async def _evaluate_async(self, conversations: List[List[Dict[str, str]]],use_tqdm=False, **kwargs) -> List[float]:
        
        ref_logprobs = await self.ref._evaluate_async(conversations, use_tqdm=use_tqdm)
        
        reward = await self.reward._evaluate_async(conversations, use_tqdm=use_tqdm)
        
        full_return = [ self.beta * ref_logprobs[i] + reward[i] for i in range(len(ref_logprobs)) ]
        
        return full_return
        
    def evaluate(self, conversations: List[List[Dict[str, str]]], use_tqdm=False, **kwargs) -> List[float]:
        return self.ref.model_client._run_on_thread_loop(
            self._evaluate_async(conversations, use_tqdm=use_tqdm, **kwargs)
        ) 
        
        
class QUEST(QAlign):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def transition_likelihood_ratio(
            self,
            previous_state: QAlign.State,
            proposal_state: QAlign.State,
            **kwargs,
        ):
            """
            Calculate the transition likelihood ratio for the Metropolis-Hastings criterion.
            """
            previous_length = list(
                map(
                    len,
                    previous_state.completion,
                )
            )
            proposal_length = list(
                map(
                    len,
                    proposal_state.completion,
                )
            )

            # Calculate the log likelihood ratios for the indices
            # old -> new
            index_log_likelihood_forward = np.array(
                [
                    log_prob_index(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        proposal_state.index,
                        previous_length,
                    )
                ]
            )

            # new -> old
            index_log_likelihood_backward = np.array(
                [
                    log_prob_index(
                        index=index,
                        truncation=n,
                    )
                    for index, n in zip(
                        proposal_state.index,
                        proposal_length,
                    )
                ],
            )

            # Calculate the log likelihood ratios for the indices
            log_likelihood_backward = index_log_likelihood_backward + previous_state.scores
            log_likelihood_forward = index_log_likelihood_forward + proposal_state.scores

            # Calculate the log transition ratio
            log_transition_ratio = log_likelihood_backward - log_likelihood_forward

            return log_transition_ratio