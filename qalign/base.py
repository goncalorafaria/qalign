import copy
import dataclasses
from typing import *

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

 
import torch
from numpy.random import randint
import math
import queue
import threading
from copy import deepcopy
import os

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def join_accepted_values(accept, proposal, state):
    return [p if a else s for a, p, s in zip(accept, proposal, state)]


def sample_index(truncation: int) -> int:
    """
    Samples a random index from 0 to truncation (exclusive), optionally in blocks.
    """
    block_size: int = 1
    return randint(0, truncation // block_size) * block_size

def log_prob_index(index: int, truncation: int) -> float:
    """
    Returns the log probability of sampling an index under a uniform discretized distribution.
    """
    block_size: int = 1
    normalization = float(truncation // block_size)
    return -math.log(normalization)
        
def process_batch_outputs(
    state_path: Any, batch_size: int
) -> List[List[Dict[str, Any]]]:
    """
    Processes batch outputs from a Quest chain into a standardized format.
    """
    outputs = []
    for i in range(batch_size):
        outputs.append(
            [
                {
                    "t": s["t"],
                    **{k: v[i] for k, v in s.items() if k != "t"},
                }
                for s in state_path
            ]
        )
    return outputs

def repeat_on_reject(output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Repeats the previous accepted state when a proposal is rejected.
    This ensures that rejected proposals don't create gaps in the output sequence.
    
    Args:
        output: List of state dictionaries from the MCMC chain
        
    Returns:
        List of state dictionaries with rejected states replaced by previous accepted states
    """
 
    processed_output = []
    last_accepted_state = None
    
    for state in output:
        # Check if this state was accepted based on the accept field
        # The accept field is a list of booleans for each chain
      
        if state["accept"]:
            # Update the last accepted state
            last_accepted_state = state["text"]
            processed_output.append(state["text"])
        else:
            processed_output.append(last_accepted_state)
            
    return processed_output


class QAlign:
    """
    This class implements the Metropolis-Hastings MCMC method with an AR transition kernel.
    The Metropolis-Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random
    samples from a probability distribution for which direct sampling is difficult. This sequence can be used to
    approximate the distribution.

    The transition kernel works the following way:
    1. Sample a token index from 0 to the sentence legth.
    2. Sample a continuation from the current state until the eos token
    """

    @dataclasses.dataclass
    class State:
        """
        This class represents the state of the Markov Chain.
        """

        reward: List[float]  # The reward obtained at the current state
        completion: List[List[float]]  # The completion tensor for the current state
        text: List[str]  # The completion text for the current state
        index: List[int]  # The index of the current state
        t: int = 0  # The current time step

        def to_json(self):
            """
            This function converts the State class to a JSON object.
            """
            return {
                "reward": self.reward,
                "completion": self.completion,
                "text": self.text,
                "t": self.t,
                "index": self.index,
            }

        def copy_relevant(self, relevant_chains: List[int]):
            return QAlign.State(
                reward=[self.reward[i] for i in relevant_chains],
                completion=[self.completion[i] for i in relevant_chains],
                text=[self.text[i] for i in relevant_chains],
                t=self.t,
                index=[self.index[i] for i in relevant_chains],
            )

        def paste_relevant(
            self,
            relevant_chains: List[int],
            additions,
        ):
            new_state = copy.deepcopy(self)

            for i, chain in enumerate(relevant_chains):
                new_state.reward[chain] = additions.reward[i]
                new_state.completion[chain] = additions.completion[i]
                new_state.text[chain] = additions.text[i]
                new_state.index[chain] = additions.index[i]

            new_state.t = additions.t

            return new_state

    @dataclasses.dataclass
    class Output:
        """
        This class represents the output of the Markov Chain.
        """
        state_path: List[Dict[str, str]]
        texts: List[Dict[str, Any]]

    def __init__(
        self, 
        model: Any,
        reward: Any,
        beta: float = 0.1,
        logratio_clamp: float = 20,
    ):
        """
        Initializes the Quest class.

        Parameters:
        - input_data (List[Dict[str, str]]): The input data for the Quest class.
        - model (LanguageModel): The language model to be used as the completion model.
        - reward (Reward): The reward model to be used for calculating the reward.
        - dist (IndexDistribution): The index distribution for sampling indices.
        - beta (float, optional): The beta value for the reward calculation. Default is 0.1.
        - logratio_clamp (int, optional): The maximum value for the log ratio. Default is 20.
        """
    
        self.model = model
        self.rm = reward

        self.beta = beta
        self.logratio_clamp = logratio_clamp
        self.steps = 0

    def compute_reward(
        self,
        proposal_text: List[str],
    ) -> List[float]:
        """
        This function calculates the reward for a given proposal text.

        Parameters:
        proposal_text (List[str]): The text for which the reward is to be calculated.
        uncomplete_indices (Union[None, List[int]], optional): Indices of uncompleted chains.

        Returns:
        List[float]: The calculated rewards.
        """
        value = self.rm.evaluate(
            proposal_text,
        )
        return value

    def get_prompt(
        self,
        input_data: List[Dict[str, str]],
    ):
        return self.model.encode(input_data)

    def draw_initial_state(self, prompt) -> State:
        """
        This function initializes the Markov chain given a prompt.

        Parameters:
        prompt: The prompt to initialize the Markov chain.

        Returns:
        State: The initial state of the Markov chain.
        """
        
        # Generate the initial completion
        completions = self.model.continuation(
            prompt,
            prefix=None,
        )

        # Decode the completion text
        completions_text = self.model.decode_tokenize(completions)
        
        state = QAlign.State(
            reward=self.compute_reward(completions_text),
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )
        
        return state

    def bootstrap_initial_state(self, prompt, samples: List[str]) -> State:
        """
        Bootstrap the initial state for the Markov chain. Setting specific samples as the start of the markov chain.

        Args:
            prompt: The prompt text.
            samples (list): List of samples.

        Returns:
            State: The initial state for the Markov chain.
        """
        completions_text = [s["completion"] for s in samples]  # list of samples.
        completions_reward = [s["reward"] for s in samples]

        completions = self.model.tokenize(completions_text)

        # Create the initial state for the Markov chain
        state = QAlign.State(
            reward=completions_reward,
            completion=completions,
            text=completions_text,
            index=[0] * len(completions),
        )

        return state

    def transition_likelihood_ratio(
        self,
        previous_state: State,
        proposal_state: State,
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
        log_likelihood_backward = index_log_likelihood_backward
        log_likelihood_forward = index_log_likelihood_forward

        # Calculate the log transition ratio
        log_transition_ratio = log_likelihood_backward - log_likelihood_forward

        return log_transition_ratio

    def transition(
        self,
        previous_state: State,
        prompt,
    ):
        """
        Generate a proposal state by sampling a new index and continuation.
        """
        completions = previous_state.completion
        indeces = [
            sample_index(
                truncation=len(completion),
            )
            for completion in completions
        ]
        if DEBUG:
            print("idx:",indeces)

        prefix = [
            completion[:index]
            for completion, index in zip(
                completions,
                indeces,
            )
        ]

        continuation_proposal = self.model.continuation(
            prompt,
            prefix,
        )

        proposal = list(
            map(
                lambda x: x[0] + x[1],
                zip(
                    prefix,
                    continuation_proposal,
                ),
            )
        )

        proposal_text = self.model.decode_tokenize(
            proposal,
        )

        proposal_state = QAlign.State(
            completion=proposal,
            reward=None,
            text=proposal_text,
            index=indeces,
            t=previous_state.t + 1,
        )

        return proposal_state

    def transition_and_evaluation(
        self,
        previous_state: State,
        prompt,
    ):
        """
        Generate a proposal state and compute its reward.
        """
        
        proposal_state = self.transition(previous_state, prompt)
        
        
        proposal_reward = self.compute_reward(
            proposal_state.text,
        )
        

        proposal_state.reward = proposal_reward

        return proposal_state

    def join_accepted_values(
        self,
        accept,
        previous_state: State,
        proposal_state: State,
    ):
        """
        Join the accepted values from the proposal state with the previous state.
        """
        # Update the state values based on the acceptance
        state = QAlign.State(
            completion=join_accepted_values(
                accept,
                proposal_state.completion,
                previous_state.completion,
            ),
            reward=join_accepted_values(
                accept,
                proposal_state.reward,
                previous_state.reward,
            ),
            text=join_accepted_values(
                accept,
                proposal_state.text,
                previous_state.text,
            ),
            index=(
                join_accepted_values(
                    accept,
                    proposal_state.index,
                    previous_state.index,
                )
                if (proposal_state.index is not None)
                else None
            ),
            t=proposal_state.t,
        )

        return state

    def start_chain(self, prompt, warm_start=None) -> State:
        """
        Start the Markov chain with an initial state.
        """
        

        if warm_start is None:
            state = self.draw_initial_state(prompt)

            # Compute the reward for the initial completion
            #for i, t in enumerate(state.text):
            #    self.samples[i].append(t)
        else:
            state = self.bootstrap_initial_state(
                prompt,
                warm_start,
            )

         

        self.stack(
            state,
            len(state.reward) * [1],
            len(state.reward) * [1.0],
        )

        return state

    def criterion(
        self, previous_state: State, proposal_state: State, **kwargs
    ) -> np.ndarray:
        """
        This function calculates the Metropolis-Hastings criterion for accepting or rejecting a proposal state.

        Parameters:
        previous_state (State): The previous state of the Markov Chain.
        proposal_state (State): The proposal state of the Markov Chain.

        Returns:
        np.ndarray: The acceptance probabilities for each proposal.
        """
        log_transition_ratio = self.transition_likelihood_ratio(
            previous_state=previous_state, proposal_state=proposal_state, **kwargs
        )

        # Calculate the log reward ratio
        log_reward_ratio = (
            np.array(proposal_state.reward) - np.array(previous_state.reward)
        ) / self.beta

        # Calculate the sum of the log transition ratio and log reward ratio
        sum_value = log_reward_ratio + log_transition_ratio

        # Clamp the sum value to avoid numerical instability
        clamped_value = np.clip(
            sum_value,
            -self.logratio_clamp,
            self.logratio_clamp,
        )

        # Calculate the detailed balance as the exponential of the clamped sum value
        detailed_balance = np.exp(clamped_value)

        # Calculate the acceptance probabilities as the minimum of the detailed balance and 1
        alpha = np.minimum(
            detailed_balance,
            np.ones_like(detailed_balance),
        )

        return alpha

    def draw_transition(
        self,
        previous_state: State,
        prompt,
    ) -> Tuple[State, np.ndarray]:
        """
        This function performs one step of the Metropolis-Hastings MCMC algorithm.
        It generates a proposal state and calculates the detailed balance to decide whether to accept or reject the proposal.

        Parameters:
        previous_state (State): The previous state of the Markov Chain.

        Returns:
        tuple: A tuple containing the proposal state and the acceptance probabilities.
        """
       

        proposal_state = self.transition_and_evaluation(
            previous_state=previous_state,
            prompt=prompt,
        )
        

        alpha = self.criterion(
            previous_state,
            proposal_state,
            prompt=prompt,
        )
        return (
            proposal_state,
            alpha,
        )

    def stack(
        self,
        state: State,
        accept: List[bool],
        alpha: List[float],
    ):
        """
        Stack the current state information for tracking.
        """
        self.state_path.append(
            {
                **state.to_json(),
                #"sample_counts": [len(s) for s in self.samples],
                "accept": accept,
                "criterion": alpha,
            }
        )
    
    def run(
        self,
        input_data: List[Dict[str, str]],
        steps: int = 100,
        warm_start: Union[None, List[str]] = None,
        use_tqdm: bool = False,
        n: Union[None, int] = None,
    ) -> Output:
        """
        This function runs the Markov Chain Monte Carlo (MCMC) method with Metropolis-Hastings algorithm.
        It iteratively draws transitions and decides whether to accept or reject them based on the detailed balance.

        Parameters:
        - steps (int): The number of steps to run the chain. Default is 100.
        - warm_start (Union[None, List[str]]): A list of warm start sentences to initialize the chain. Default is None.
        - use_tqdm (bool): Whether to use tqdm for progress bar. Default is False.
        - n (Union[None, int]): The number of iterations to run the chain. Default is None, which is equal to the number of steps.

        Returns:
        - Output: A named tuple containing the state path.
        """
        
        self.state_path = []
        
        if n is None:
            n = steps

        self.steps = steps
        # Draw the initial state
        self.prompt = self.get_prompt(input_data)

        context = [self.model.get_prompt(**data) for data in input_data]
        self.rm.set_context(context)
                
        state = self.start_chain(
            self.prompt,
            warm_start=warm_start,
        )

        if use_tqdm:
            # We mod by 20 to avoid having too many progress bars on screen
            unique_position = id(self) // 100 % 5

            # Create a short unique identifier for the description
            unique_id = hex(id(self))[-5:]  # Last 6 characters of hex ID

            iter = tqdm(
                range(n),
                desc=f"Chain {unique_id}",
                position=unique_position,
                leave=True,
            )
        else:
            iter = range(n)

        # Run the chain for the specified number of steps
        for i in iter:
            prompt = self.prompt

            proposal_state, A = self.draw_transition(
                previous_state=state,
                prompt=prompt,
            )

           
            # Decide whether to accept the proposal
            accept = np.array(
                bernoulli(A).rvs(),
            ).reshape(A.shape)
            
            if DEBUG:
                print("accept:",accept)
                print("--"*20)

            state = self.join_accepted_values(
                accept=accept,
                previous_state=state,
                proposal_state=proposal_state,
            )

            self.stack(
                proposal_state,
                accept.tolist(),
                A.tolist(),
            )

             

         
        
        outputs=process_batch_outputs(self.state_path, len(self.prompt))
        dupped_outputs= [ {"input":ind, "outputs": repeat_on_reject(output)} for ind,output in zip(input_data,outputs)]
        return QAlign.Output(
            state_path=self.state_path,
            texts=dupped_outputs
        )



    def run_pipelined(
        self, 
        data_batches,
        steps,
        workers=0,
        use_tqdm=False,
        **kwargs,
    ):
        if workers==0:
            return self.run(
                input_data=data_batches,
                steps=steps,
                use_tqdm=use_tqdm,
                **kwargs,
            ).texts
        
        batch_queue = queue.Queue(maxsize=workers)  # Limit queue size to control memory
        result_queue = queue.PriorityQueue() 
        
        # Dictionary to store results in order
        results_dict = OrderedDict()
        
        # Event to signal threads to stop
        stop_event = threading.Event()
        
        chain =None
        def worker_thread(thread_id):
            
        
            """Worker thread that processes batches"""
            while not stop_event.is_set():
                try:
                    # Get batch from queue with timeout
                    batch_index, data_batch = batch_queue.get(timeout=1)
                    
                    chain_outputs = deepcopy(chain).run(
                        input_data=data_batch,
                        steps=steps,
                        use_tqdm=use_tqdm,
                    )
                    
                    outputs = process_batch_outputs(chain_outputs, len(data_batch))
                    
                    # Put result in priority queue with batch index as priority
                    result_queue.put((batch_index, (data_batch, outputs)))
                    
                    batch_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in thread {thread_id}: {e}")
                    # Put error in result queue to handle it properly
                    result_queue.put((batch_index, (data_batch, None, e)))
                    batch_queue.task_done()
        
        # Start worker threads
        threads = []
        for i in range(workers):
            t = threading.Thread(target=worker_thread, args=(i,))
            t.start()
            threads.append(t)
        
        # Producer: Add batches to queue
        def producer():
            for i, data_batch in enumerate(data_batches):
                batch_queue.put((i, data_batch))
        
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        
        # Collector: Process results in order
        total_batches = len(data_batches)
        processed_count = 0
        next_expected_index = 0
        
        total_results_packed=[]
        while processed_count < total_batches:
            try:
                # Get result from queue
                batch_index, result = result_queue.get(timeout=5)
                
                # Store result
                results_dict[batch_index] = result
                
                # Process results in order
                while next_expected_index in results_dict:
                    data_batch, outputs, *error = results_dict[next_expected_index]
                    
                    if error:
                        raise error[0]
                    
                    total_results_packed.extend(
                        zip(
                            data_batch,
                            outputs,
                        )
                    )
                    
                    # Remove processed result
                    del results_dict[next_expected_index]
                    next_expected_index += 1
                    processed_count += 1
                    
            except queue.Empty:
                # Check if threads are still alive
                if not any(t.is_alive() for t in threads):
                    break
        
        # Wait for producer to finish
        producer_thread.join()
        
        # Wait for all batches to be processed
        batch_queue.join()
        
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish
        for t in threads:
            t.join()
        
        return total_results_packed