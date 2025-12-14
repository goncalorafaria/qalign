import copy
import dataclasses
from typing import *
import time
import functools

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

 
import torch

import queue
import threading
from copy import deepcopy
import os
from qalign.utils.timing import timing_decorator, TimingStats
from qalign.utils.list import join_accepted_values, process_batch_outputs, repeat_on_reject
from qalign.utils.math import sample_index, log_prob_index

DEBUG = os.getenv("DEBUG", "False").lower() == "true"


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
        
        # Initialize timing statistics
        self.timing_stats = TimingStats()

    @timing_decorator('compute_reward')
    def compute_reward(
        self,
        text: List[str],
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
            text,
        )
        
        return value

    def get_prompt(
        self,
        input_data: List[Dict[str, str]],
    ):
        return self.model.encode(input_data)

    def draw_initial_state(self, prompt, conversations) -> State:
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
        

        rewards = self.compute_reward(
            [ c + [{"role":"assistant", "content":t}] for c,t in zip(conversations,completions_text)],
        )
        
        state = QAlign.State(
            reward=rewards,
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

    @timing_decorator('transition')
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
        
        
        proposal_state.reward = self.compute_reward(
            [p + c for p, c in zip(self.prompts,proposal_state.text)],
        )
        


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

    @timing_decorator('start_chain')
    def start_chain(self, prompt, conversations, warm_start=None) -> State:
        """
        Start the Markov chain with an initial state.
        """
        

        if warm_start is None:
            state = self.draw_initial_state(prompt, conversations)

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
        conversations: List[ List[Dict[str, str]] ],
        steps: int = 100,
        warm_start: Union[None, List[str]] = None,
        use_tqdm: bool = False,
        callbacks: List[Callable] = [],
        tqdm_index: int = 0,
        tqdm_total: int = 1,
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
        
        prompts = self.model.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        
        self.state_path = []


        self.steps = steps
        # Draw the initial state
        self.prompts = prompts
        prompt_ids = self.get_prompt(prompts)
        #self.rm.set_context(prompts)
                
        state = self.start_chain(
            prompt_ids,
            conversations,
            warm_start=warm_start,
        )
        for callback in callbacks:
            callback(state)
            
        if use_tqdm:
            # We mod by 20 to avoid having too many progress bars on screen
            unique_position = tqdm_index % tqdm_total

            # Create a short unique identifier for the description
            #unique_id = hex(id(self))[-5:]  # Last 6 characters of hex ID

            iter = tqdm(
                list(range(self.steps)),
                desc=f"Chain {tqdm_index}",
                position=unique_position,
                leave=True,
            )
        else:
            iter = list(range(self.steps))

        # Run the chain for the specified number of steps
        for i in iter:

            proposal_state = self.transition(state, prompt_ids)

            
            proposal_state.reward = self.compute_reward(
                [ c + [{"role":"assistant", "content":t}] for c,t in zip(conversations,proposal_state.text)],
            )

            A = self.criterion(
                state,
                proposal_state,
                prompt=prompt_ids,
            )
           
            # Decide whether to accept the proposal
            accept = np.array(
                bernoulli(A).rvs(),
            ).reshape(A.shape)
            

            state = self.join_accepted_values(
                accept=accept,
                previous_state=state,
                proposal_state=proposal_state,
            )
            
            for callback in callbacks:
                callback(state)

            self.stack(
                proposal_state,
                accept.tolist(),
                A.tolist(),
            )         
        
        outputs=process_batch_outputs(self.state_path, len(prompts))
        dupped_outputs= [ {"input":ind, "outputs": repeat_on_reject(output)} for ind,output in zip(prompts,outputs)]
        
        # Print timing statistics if DEBUG is enabled
        if DEBUG:
            self.print_timing_stats()
        
        return QAlign.Output(
            state_path=outputs,
            texts=dupped_outputs,
        )



    def run_pipelined(
        self, 
        conversations,
        steps,
        workers=0,
        use_tqdm=False,
        batch_size=16,
        **kwargs,
    ):
        if workers==0:
            outputs = self.run(
                conversations=conversations,
                steps=steps,
                use_tqdm=use_tqdm,
                **kwargs,
            )
            
            return list(zip(conversations, outputs.state_path))
        
        batch_queue = queue.Queue(maxsize=workers)  # Limit queue size to control memory
        result_queue = queue.PriorityQueue() 
        
        # Dictionary to store results in order
        results_dict = OrderedDict()
        
        # Event to signal threads to stop
        stop_event = threading.Event()
        
        chain = self
        
        print(f"Conversations: {len(conversations)}")
        #print(conversations[0])
        def worker_thread(thread_id):
            """Worker thread that processes batches"""
            while not stop_event.is_set():
                try:
                    # Get batch from queue with timeout
                    batch_index, data_batch = batch_queue.get(timeout=1)
                    
                    
                    print(f"Processing batch {batch_index} with {len(data_batch)} conversations - {thread_id}")
                    #print(data_batch[0])
                    chain_outputs = deepcopy(chain).run(
                        conversations=data_batch,
                        steps=steps,
                        use_tqdm=use_tqdm,
                        tqdm_index=thread_id,
                        tqdm_total=workers,
                    )
                    
                    outputs = chain_outputs.state_path
                    
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
            ## break in to batches of 16
            batch_idx = 0
            for i in range(0, len(conversations), batch_size):
                batch_queue.put((batch_idx, conversations[i:i+batch_size]))
                batch_idx += 1
            
            #for i, data_batch in enumerate(conversations):
            #    batch_queue.put((i, data_batch))
        
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        
        # Collector: Process results in order
        total_batches = (len(conversations) + batch_size - 1) // batch_size  # Calculate number of batches
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
    
    def print_timing_stats(self):
        """
        Print timing statistics for all recorded operations.
        """
        if not self.timing_stats.operations:
            print("No timing statistics recorded.")
            return
        
        for operation, stats in sorted(self.timing_stats.operations.items()):
            print(f"\n=== {operation.replace('_', ' ').title()} Timing Statistics ===")
            print(f"Total calls: {stats['count']}")
            print(f"Average time: {stats['mean']:.6f} seconds")
            print(f"Total time: {stats['total']:.6f} seconds")
        
        print("=" * 50)
            
            
    def __str__(self):
        return f"QAlign(model={self.model}, reward={self.rm})"