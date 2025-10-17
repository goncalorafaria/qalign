import os

import multiprocessing as mp


from typing import *


from tqdm import tqdm

import numpy as np

## qalign
from qalign.utils.ifeval.eval import test_instruction_following_loose
from qalign.utils.math import get_last_math, get_last_option, get_last_number

## quest
#from qalign.reward import ValueHead
from qalign.reward import RemoteReward
from qalign.utils.list import (
    flatten_list,
    unflatten_list,
    get_unique_mapping,
    invert_unique_mapping,
    chunked,
)
from qalign.reward import (
    Reward,
    #RewardModel,
    #ContextualRewardModel,
)

## expkit
from expkit import (
    ExpSetup,
    Exp,
    Evalutor,
)
from expkit.ops import proj


def compress_predictions(instances, n=None):
    prompts, tokens, vocabs, counts = (
        [],
        [],
        [],
        [],
    )

    # instances = experiment.instances()
    if n is not None:
        instances = instances[:n]

    for i in instances:

        tksi, vcbi = get_unique_mapping(
            list(
                map(
                    proj("text"),
                    i["outputs"],
                )
            )
        )

        promptsi = [i["input"]["chat_template_prompt"]] * len(vcbi)

        prompts.append(promptsi)
        vocabs.append(vcbi)
        tokens.append(tksi)
        counts.append(len(vcbi))

    return prompts, tokens, vocabs, counts


class ExactEval(Evalutor):

    def __init__(self, process_fn, name="exact"):
        super().__init__(name)

        self.process_fn = process_fn

    def eval(self, experiment: Exp):

        results = []

        instances = experiment.instances()

        if not isinstance(instances, list):
            instances = [instances]

        for i in tqdm(instances):

            preds = [self.process_fn(x["text"]) for x in i["outputs"]]

            target = self.process_fn(i["input"]["answer"])

            scores = list(
                map(
                    lambda p: int(p == target),
                    preds,
                )
            )

            results.append({"scores": scores})

        return results


class IFEval(Evalutor):

    def __init__(self, name="ifeval"):
        super().__init__(name)

    def create_eval_pairs(self, experiment):
        """Create all input-output pairs for evaluation."""
        instances = experiment.instances()
        if not isinstance(instances, list):
            instances = [instances]

        eval_pairs = []
        for idx, instance in enumerate(instances):
            for output in instance["outputs"]:
                eval_pairs.append(
                    {
                        "instance_idx": idx,
                        "input": instance["input"],
                        "output": {
                            "text": output["text"].replace("<|end_of_text|>", "")
                        },
                    }
                )
        return eval_pairs, len(instances)

    def process_single_pair(self, pair):
        """Process a single input-output pair."""

        return {
            "instance_idx": pair["instance_idx"],
            "score": test_instruction_following_loose(pair["input"], pair["output"])[
                "follow_all_instructions"
            ],
        }

    def eval(self, experiment: Exp):
        # Step 1: Create all evaluation pairs
        eval_pairs, num_instances = self.create_eval_pairs(experiment)

        # Step 2: Process pairs in parallel
        with mp.Pool() as pool:
            results_flat = list(
                tqdm(  # pool.i
                    pool.map(self.process_single_pair, eval_pairs),
                    total=len(eval_pairs),
                    desc="Processing pairs",
                )
            )

        # Step 3: Reorganize results into original structure
        results = []
        for i in range(num_instances):
            instance_scores = [
                r["score"] for r in results_flat if r["instance_idx"] == i
            ]
            results.append({"scores": instance_scores})

        return results


class ExactLastNumberEval(ExactEval):

    def __init__(self):
        super().__init__(
            process_fn=get_last_number,
            name="lastnumber",
        )


class ExactMATHEval(ExactEval):

    def __init__(self):
        super().__init__(
            process_fn=get_last_math,
            name="lastmath",
        )


class ExactQAEval(ExactEval):

    def __init__(self):
        super().__init__(
            process_fn=get_last_option,
            name="lastoption",
        )


class RewardEval(Evalutor):

    def __init__(self, reward: Reward, n=None, chunk_size=128):
        super().__init__(reward.get_name())
        self.reward = reward
        self.n = n
        self.chunk_size = chunk_size

    def eval(self, experiment: Exp):

        all_duplicated_scores = []

        for instance_chunk in chunked(
            experiment.instances(lazy_iterable=True), self.chunk_size
        ):
            # Process current chunk of instances
            prompts, tokens, vocabs, counts = compress_predictions(
                instance_chunk, n=self.n
            )
            
            #  instance_chunk[0]['input']["chat_template_prompt"]
            
            # rewards = self.compute_reward(
            #[ c + [{"role":"assistant", "content":t}] for c,t in zip(conversations,completions_text)],
            # )#
            
            
            chunk_flat_vocabs = flatten_list(vocabs)
            chunk_flat_prompts = flatten_list(prompts)
            
            conversations = [ [c, {"role":"assistant", "content":t}] for c,t in zip(chunk_flat_prompts,chunk_flat_vocabs)]
            
            # Evaluate just this chunk's candidates
            
            chunk_scores = self.reward.evaluate(
                conversations=conversations,
                use_tqdm=True,  # Disable per-chunk progress bars
            )

            # Process scores for this chunk only
            chunk_unflattened = unflatten_list(chunk_scores, counts)
            all_duplicated_scores.extend(
                [
                    {"scores": invert_unique_mapping(tks, rsi)}
                    for tks, rsi in zip(tokens, chunk_unflattened)
                ]
            )

            del instance_chunk
            del prompts
            del tokens
            del vocabs
            del counts

        return all_duplicated_scores
