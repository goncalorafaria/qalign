from collections import Counter
import re
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


from typing import *

from transformers import AutoTokenizer

## quest
from qalign.utils.list import chunked

## qalign
from qalign.utils.mbr import mbr_pick_progression, weighted_mbr_pick_progression
from qalign.utils.math import (
    get_last_number,
    get_last_math,
    get_last_option,
    generate_axis,
)

## expkit
from expkit import (
    Exp,
    Evalutor,
)


def get_strategy(
    strategy: str,
    key: str,
    p=None,
    beta=1.0,
    c=1.0,
    gaps=64,
    exp_rate=False,
    **kwargs,
) -> Evalutor:

    if strategy == "voting":
        pick = VotingPick(gaps=gaps, exp_rate=exp_rate, multiprocessing=p, **kwargs)

    elif strategy == "bon":
        pick = BonPick(
            key=key, gaps=gaps, exp_rate=exp_rate, multiprocessing=p, **kwargs
        )

        # setup = setup.filter(lambda x: x.has_eval(key))

    elif strategy == "weighted-voting":
        pick = WeightedVotingPick(
            key=key,
            gaps=gaps,
            exp_rate=exp_rate,
            multiprocessing=p,
            beta=beta,
            **kwargs,
        )
        # setup = setup.filter(lambda x: x.has_eval(key))
    elif strategy == "last":
        pick = LastPick(gaps=gaps, exp_rate=exp_rate, multiprocessing=p, **kwargs)
    elif strategy == "mbr":
        pick = MBRPick(gaps=gaps, **kwargs)
    elif strategy == "wmbr":
        pick = WMBRPick(gaps=gaps, key=key, exp_rate=exp_rate, **kwargs)
    elif strategy == "bonresample":
        pick = BonResamplePick(
            key=key, gaps=gaps, exp_rate=exp_rate, multiprocessing=p, **kwargs
        )
    else:
        raise ValueError("Invalid strategy")

    return pick


def get_param_count(exp):
    # example : model_path = "meta-llama/Llama-3.1-8B-Instruct"
    # or model_path = "allenai/olmo-2-2b-it"
    # I need to look for B or b and then get the number before it

    # get the number before B or b
    # get the number before B or b

    model_path = exp.meta["model_path"]
    match = re.search(r"(\d+)[Bb]", model_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No match found")


def get_flops_by_scores(exp, key="", strategy="voting"):

    with Pool() as p:
        eval_result = get_strategy(
            strategy=strategy,
            p=p,
            exp_rate=True,
            gaps=2,
            key=key,
        ).eval(exp)

    axis = eval_result[0]["axis"]

    tokenizer = AutoTokenizer.from_pretrained(exp.meta["model_path"])
    token_counts = []
    for instance in exp.instances():

        if exp.meta["variant"] == "ancestral":

            ids = tokenizer([o["text"] for o in instance["outputs"]], padding=False)

            token_counts.append(list(map(len, ids["input_ids"])))
        else:

            token_counts.append(
                [len(o["completion"][o["index"] :]) for o in instance["outputs"]]
            )
        # all_criteria.append(x)
    token_counts = np.array(token_counts)

    # sample a few points from axis ...

    D_test = []
    for i in axis:
        D_test.append(np.sum(token_counts[:, :i], axis=1).mean(0))

    ## billion in scientific notation
    ##
    IFLOPS = [
        2 * 2 * float(get_param_count(exp)) * v for v in D_test
    ]  # one extra 2 for reward.
    return {
        "IFLOPS": IFLOPS,
        "D_test": D_test,
        "scores": eval_result[0]["scores"],
    }


def get_flops_prev(exp, axis):

    tokenizer = AutoTokenizer.from_pretrained(exp.meta["model_path"])
    token_counts = []
    for instance in exp.instances():

        if exp.meta["variant"] == "ancestral":

            ids = tokenizer([o["text"] for o in instance["outputs"]], padding=False)

            token_counts.append(list(map(len, ids["input_ids"])))
        else:

            token_counts.append(
                [len(o["completion"][o["index"] :]) for o in instance["outputs"]]
            )
        # all_criteria.append(x)
    token_counts = np.array(token_counts)

    # sample a few points from axis ...

    D_test = []
    for i in axis:
        D_test.append(np.sum(token_counts[:, :i], axis=1).mean(0))

    ## billion in scientific notation
    ##
    IFLOPS = [
        2 * float(get_param_count(exp)) * v for v in D_test
    ]  # one extra 2 for reward.
    return {
        "IFLOPS": IFLOPS,
        "D_test": D_test,
    }


def real_avg_token_counts(exp, sweep, k=64):
    tokenizer = AutoTokenizer.from_pretrained(exp.meta["model_path"])
    token_counts = []

    instances = exp.instances(lazy_iterable=True)

    for i, instance in enumerate(
        tqdm(instances, desc="Processing Instances", total=k + 1)
    ):

        if exp.meta["variant"] == "ancestral":

            ids = tokenizer([o["text"] for o in instance["outputs"]], padding=False)

            token_counts.append(list(map(len, ids["input_ids"])))
        else:

            token_counts.append(
                [len(o["completion"][o["index"] :]) for o in instance["outputs"]]
            )

        if i > k:
            break

    token_counts = np.array(token_counts)

    avg_token_counts = np.mean(token_counts, axis=0)

    if exp.meta["variant"] == "ancestral":
        if ("bon" in sweep) or ("weighted" in sweep) or ("wmbr" in sweep):
            avg_token_counts *= 2

    else:  # questrlhf
        avg_token_counts *= 2

    return avg_token_counts


def fake_avg_token_counts(exp, sweep, d=500):
    avg_token_counts = np.array([d] * exp.get("steps"))  # * n

    if "DPO" in exp.meta["model_path"]:
        avg_token_counts = avg_token_counts * 1.5

    else:

        # sample a few points from axis ...
        if exp.meta["variant"] == "ancestral":
            if ("bon" in sweep) or ("weighted" in sweep) or ("wmbr" in sweep):
                avg_token_counts *= 2

        else:  # questrlhf
            avg_token_counts[0] *= 2
            # avg_token_counts[1:] = avg_token_counts[1:] /

    return avg_token_counts


def get_flops(exp, axis, sweep, n=None, fake=True):

    # tokenizer = AutoTokenizer.from_pretrained(exp.meta["model_path"])

    if n is None:
        n = exp.meta["n"]

    if fake:
        avg_token_counts = fake_avg_token_counts(exp, sweep=sweep)
    else:
        avg_token_counts = real_avg_token_counts(exp, sweep=sweep)

    D_test = []
    for i in axis:
        D_test.append(np.sum(avg_token_counts[:i]))

    ## billion in scientific notation
    ##
    IFLOPS = [
        2 * float(get_param_count(exp)) * v for v in D_test
    ]  # one extra 2 for reward.
    return {
        "IFLOPS": IFLOPS,
        "D_test": D_test,
    }


def softmax(x, temp=1.0):
    x = np.array(x)
    e_x = np.exp(x / temp)
    return e_x / e_x.sum()


def log_softmax(x, temp=1.0):
    x = np.array(x)
    x = x / temp
    x_max = np.max(x)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max))) + x_max
    return x - log_sum_exp


def is_weighted_mode_correct(
    instance_and_score, n=512, beta=1.0, extract_func=get_last_number
):
    instance, scores = instance_and_score

    r = scores[:n]
    p = softmax(r, temp=beta)
    answer = extract_func(instance["input"]["answer"])
    outputs = instance["outputs"][:n]

    responses = [
        (extract_func(output["text"]), float(pi)) for output, pi in zip(outputs, p)
    ]

    response_dict = {}
    for response, prob in responses:
        if response in response_dict:
            response_dict[response] += prob
        else:
            response_dict[response] = prob

    if answer in response_dict:
        correct_estimate = response_dict[answer]
    else:
        correct_estimate = 0

    max_response = max(response_dict, key=response_dict.get)

    return int(max_response == answer), correct_estimate


def get_last(instance, n=512, extract_func=get_last_number):

    outputs = instance["outputs"][:n]
    responses = [extract_func(output["text"]) for output in outputs]
    answer = extract_func(instance["input"]["answer"])

    return int(responses[-1] == answer), 0.0


def is_mode_correct(instance, n=512, burnin=0, extract_func=get_last_number):

    outputs = instance["outputs"][burnin:n]

    answer = extract_func(instance["input"]["answer"])

    responses = [extract_func(output["text"]) for output in outputs]

    # compute the most common element in accepted

    c = Counter(responses)

    if answer in c:
        correct_estimate = c[answer] / len(responses)

    else:
        correct_estimate = 0

    if len(c) == 0:
        most_common = "NaN"
    else:
        most_common = c.most_common(1)[0][0]

    return int(most_common == answer), correct_estimate


def is_max_correct(instance_and_score, n=512):
    scores, gt = instance_and_score
    max_index = np.argmax(scores["scores"][:n])

    return gt["scores"][max_index]


#    max_response = extract_func(instance["outputs"][max_index]["text"])
#    # compute the most common element in accepted
#    return int(max_response == extract_func(instance["input"]["answer"]))


def is_resample_correct(
    instance_and_score, n=512, extract_func=get_last_number, r=128, trials=16
):
    instance, scores = instance_and_score
    outputs = instance["outputs"][:n]
    scores = np.array(scores[:n])
    answer = extract_func(instance["input"]["answer"])

    def shuffle(x):
        np.random.shuffle(x)
        return x

    # sample 16 times a sublist of scores of size "r"

    inds = [shuffle(np.arange(n))[:r].tolist() for _ in range(trials)]

    pick_ind = [indsi[np.argmax(scores[indsi])] for indsi in inds]
    responses = [extract_func(outputs[i]["text"]) for i in pick_ind]

    c = Counter(responses)

    if answer in c:
        correct_estimate = c[answer] / len(responses)

    else:
        correct_estimate = 0

    if len(c) == 0:
        most_common = "NaN"
    else:
        most_common = c.most_common(1)[0][0]

    return int(most_common == answer)


def join_instances_and_repeat_accepts(instances, is_quest=True):
    return _join_instances_and_repeat_accepts(instances, is_quest=is_quest)[0]


def join_instances_and_repeat_accepts_with_scores(
    instances,
    evals_scores,
    is_quest=True,
):
    return _join_instances_and_repeat_accepts(
        instances, is_quest=is_quest, evals_scores=evals_scores
    )


def _join_instances_and_repeat_accepts(instances, is_quest=True, evals_scores=None):
    inputs = instances[0]["input"]
    outputs = []
    scores = []

    if is_quest:
        outputs_cat = []
        scores_cat = []
        for i, es in zip(instances, evals_scores or [None] * len(instances)):
            accepted_outputs = []
            accepted_scores = []
            last_accepted = None
            for output, s in zip(
                i["outputs"], es["scores"] if es else [None] * len(i["outputs"])
            ):
                if output["accept"]:
                    accepted_outputs.append(output)
                    if es:
                        accepted_scores.append(s)
                    last_accepted = (output, s)
                elif last_accepted is not None:
                    accepted_outputs.append(last_accepted[0])
                    if es:
                        accepted_scores.append(last_accepted[1])

            outputs_cat.append(accepted_outputs)
            if es:
                scores_cat.append(accepted_scores)

        for i in range(len(outputs_cat[0])):
            for o in outputs_cat:
                outputs.append(o[i])
            if evals_scores:
                for s in scores_cat:
                    scores.append(s[i])
    else:
        outputs = [j for i in instances for j in i["outputs"]]
        if evals_scores:
            scores = [s for es in evals_scores for s in es["scores"]]

    # print(len(outputs))
    return {"input": inputs, "outputs": outputs}, scores


class VotingPick(Evalutor):

    def __init__(
        self,
        gaps=2,
        exp_rate=True,
        max_num_chains=None,
        multiprocessing=None,
        burnin=0,
        extract="lastnumber",
        n=None,
        **kwargs,
    ):

        super().__init__(
            extract + "-voting" if n is None else extract + "-voting-" + str(n)
        )
        self.gaps = gaps
        self.pmap = multiprocessing.map if multiprocessing else map
        self.exp_rate = exp_rate
        self.max_num_chains = max_num_chains
        self.burnin = burnin
        self.n = n
        self.chunk_size = 64

        if extract == "lastnumber":
            self.extract = get_last_number
        elif extract == "lastmath":
            self.extract = get_last_math

        elif extract == "lastoption":
            self.extract = get_last_option
        else:
            ValueError("Invalid extract function")

    def eval(self, exp: Exp):

        if self.gaps > 0:
            if not self.exp_rate:
                x_axis = generate_axis(exp.meta["steps"] + 1, self.gaps)
            else:
                x_axis = generate_axis(exp.meta["steps"] + 1, exp.meta["steps"])

        else:
            x_axis = [exp.meta["steps"]]

        # num_chains = exp.meta.get("num_chains", 1)
        # max_num_chains = self.max_num_chains if self.max_num_chains else num_chains

        # instances are in groups of num_chains .. i.e.
        # [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
        # I want to group into :
        # [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
        # so that I can compute the mean of the estimates
        # for each group

        accs_full = {xi: [] for xi in x_axis}

        c = 0
        for instance_chunk in tqdm(
            chunked(exp.instances(lazy_iterable=True), self.chunk_size)
        ):

            c += len(instance_chunk)

            if c > self.n:
                break

            is_quest = "quest" in exp.meta["variant"]

            # instances = total_instances

            instances = [
                join_instances_and_repeat_accepts([i], is_quest=is_quest)
                for i in instance_chunk
            ]
            # print(x_axis)
            # print(len(instances))

            for i in x_axis:

                results = self.pmap(
                    partial(
                        is_mode_correct,
                        n=i * 1,
                        burnin=0,
                        extract_func=self.extract,
                    ),
                    instances,
                )

                accs, estimates = zip(*results)

                accs_full[i].extend(accs)

                # scores.append({"axis": x_axis, "scores": results})
                # x_results.append(float(np.mean(list((accs)))))
                # n_estimates.append((estimates))
                # accs_full.append(accs)

            # final_estimate = n_estimates[-1]
            # estimate = [float(np.mean(np.abs(i - final_estimate))) for i in n_estimates]

        results = [
            {
                "axis": x_axis,
                "scores": [float(np.mean(accs_full[i])) for i in x_axis],
                "accs": [accs_full[i] for i in x_axis],
            }
        ]
        return results


class LastPick(Evalutor):

    def __init__(
        self,
        gaps=2,
        exp_rate=True,
        max_num_chains=None,
        multiprocessing=None,
        extract="lastnumber",
        n=None,
        **kwargs,
    ):
        super().__init__("last-pick" if n is None else "last-pick-" + str(n))
        self.gaps = gaps
        self.pmap = multiprocessing.map if multiprocessing else map
        self.exp_rate = exp_rate
        self.max_num_chains = max_num_chains
        self.n = n
        self.extract_key = extract

        if extract == "lastnumber":
            self.extract = get_last_number
        elif extract == "lastmath":
            self.extract = get_last_math
        elif extract == "lastoption":
            self.extract = get_last_option
        else:
            ValueError("Invalid extract function")

    def eval(self, exp: Exp):

        if self.gaps > 0:
            if not self.exp_rate:
                x_axis = generate_axis(exp.meta["steps"] + 1, self.gaps)
            else:
                x_axis = generate_axis(exp.meta["steps"] + 1, exp.meta["steps"])

        else:
            x_axis = [exp.meta["steps"]]

        num_chains = exp.meta.get("num_chains", 1)
        max_num_chains = self.max_num_chains if self.max_num_chains else num_chains

        # instances are in groups of num_chains .. i.e.
        # [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
        # I want to group into :
        # [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
        # so that I can compute the mean of the estimates
        # for each group

        is_quest = "quest" in exp.meta["variant"]

        instances = [
            join_instances_and_repeat_accepts(
                instances[i : i + num_chains][:max_num_chains], is_quest=is_quest
            )
            for i in range(0, len(instances), num_chains)
        ]
        # print(x_axis)
        # print(len(instances))

        x_results = []
        n_estimates = []
        accs_full = []
        for i in tqdm(x_axis):
            is_quest = "quest" in exp.meta["variant"]

            results = self.pmap(
                partial(
                    get_last,
                    n=i * max_num_chains,
                    extract_func=self.extract,
                ),
                instances,
            )

            accs, estimates = zip(*results)

            # scores.append({"axis": x_axis, "scores": results})
            x_results.append(float(np.mean(list((accs)))))
            n_estimates.append((estimates))
            accs_full.append(accs)

        # final_estimate = n_estimates[-1]
        # estimate = [float(np.mean(np.abs(i - final_estimate))) for i in n_estimates]

        return [
            {
                "axis": x_axis,
                "scores": x_results,
                "estimates": n_estimates,
                "accs": accs_full,
            }
        ]


class BonPick(Evalutor):

    def __init__(
        self,
        key,
        gaps=2,
        exp_rate=True,
        multiprocessing=None,
        extract="lastnumber",
        n=None,
        **kwargs,
    ):
        super().__init__(key + "-bon" if n is None else key + "-bon-" + str(n))
        self.reward_key = key
        self.gaps = gaps
        self.pmap = multiprocessing.map if multiprocessing else map
        self.exp_rate = exp_rate
        self.n = n
        self.extract_key = extract

    def eval(self, exp: Exp):

        assert exp.has_eval(
            self.reward_key
        ), f"Experiment does not have {self.reward_key} eval"

        if self.gaps > 0:
            if not self.exp_rate:
                x_axis = generate_axis(exp.meta["steps"] + 1, self.gaps)
            else:
                x_axis = generate_axis(exp.meta["steps"] + 1, exp.meta["steps"])

        else:
            x_axis = [exp.meta["steps"]]

        # instances = exp.instances()

        if self.n is not None:
            n = self.n
        else:
            n = exp.meta["n"]
        #    instances = instances[: self.n]

        x_results = []
        for i in tqdm(x_axis):
            # spawn pool of workers

            results = list(
                self.pmap(
                    partial(
                        is_max_correct,
                        n=i,
                    ),
                    zip(
                        exp.get_eval(self.reward_key)[:n],
                        exp.get_eval(self.extract_key)[:n],
                    ),
                )
            )

            # scores.append({"axis": x_axis, "scores": results})
            x_results.append(float(np.mean(list((results)))))

        return [{"axis": x_axis, "scores": x_results}]


class BonResamplePick(Evalutor):

    def __init__(
        self,
        key,
        gaps=2,
        exp_rate=True,
        multiprocessing=None,
        r=128,
        trials=32,
        extract="lastnumber",
        n=None,
        **kwargs,
    ):
        super().__init__(
            key + "-bonresample-" + str(r) + "-" + str(trials)
            if n is None
            else key + "-bonresample-" + str(r) + "-" + str(trials) + "-" + str(n)
        )
        self.reward_key = key
        self.gaps = gaps
        self.pmap = multiprocessing.map if multiprocessing else map
        self.exp_rate = exp_rate
        self.r = r
        self.trials = trials
        self.n = n

        if extract == "lastnumber":
            self.extract = get_last_number
        elif extract == "lastmath":
            self.extract = get_last_math
        elif extract == "lastoption":
            self.extract = get_last_option
        else:
            ValueError("Invalid extract function")

    def eval(self, exp: Exp):

        assert exp.has_eval(
            self.reward_key
        ), f"Experiment does not have {self.reward_key} eval"

        if self.gaps > 0:
            if not self.exp_rate:
                x_axis = generate_axis(exp.meta["steps"] + 1, self.gaps)
            else:
                x_axis = generate_axis(exp.meta["steps"] + 1, exp.meta["steps"])

        else:
            x_axis = [exp.meta["steps"]]

        is_quest = "quest" in exp.meta["variant"]
        num_chains = exp.meta.get("num_chains", 1)
        instances = exp.instances()
        if self.n is not None:
            instances = instances[: self.n]
        evals_scores = exp.get_eval(self.reward_key)
        instances_and_scores = [
            join_instances_and_repeat_accepts_with_scores(
                instances[i : i + num_chains],
                evals_scores=evals_scores[i : i + num_chains],
                is_quest=is_quest,
            )
            for i in range(0, len(instances), num_chains)
        ]

        x_results = []
        for i in tqdm(x_axis):
            # spawn pool of workers

            results = list(
                self.pmap(
                    partial(
                        is_resample_correct,
                        n=i,
                        extract_func=self.extract,
                        r=self.r,
                        trials=self.trials,
                    ),
                    instances_and_scores,
                )
            )

            # scores.append({"axis": x_axis, "scores": results})
            x_results.append(float(np.mean(list((results)))))

        return [{"axis": x_axis, "scores": x_results}]


class WeightedVotingPick(Evalutor):

    def __init__(
        self,
        key,
        gaps=2,
        exp_rate=False,
        multiprocessing=False,
        beta=1.0,
        extract="lastnumber",
        n=None,
        **kwargs,
    ):
        super().__init__(
            key + "-weighted-voting-" + str(beta).replace(".", ";")
            if n is None
            else key + "-weighted-voting-" + str(beta).replace(".", ";") + "-" + str(n)
        )
        self.reward_key = key
        self.gaps = gaps
        self.exp_rate = exp_rate
        self.pmap = multiprocessing.map if multiprocessing else map
        self.beta = beta
        self.n = n
        self.chunk_size = 64

        if extract == "lastnumber":
            self.extract = get_last_number
        elif extract == "lastmath":
            self.extract = get_last_math
        elif extract == "lastoption":
            self.extract = get_last_option
        else:
            ValueError("Invalid extract function")

    def eval(self, exp: Exp):

        assert exp.has_eval(
            self.reward_key
        ), f"Experiment does not have {self.reward_key} eval"

        if self.gaps > 0:
            if not self.exp_rate:
                x_axis = generate_axis(exp.meta["steps"] + 1, self.gaps)
            else:
                x_axis = generate_axis(exp.meta["steps"] + 1, exp.meta["steps"])

        else:
            x_axis = [exp.meta["steps"]]

        accs_full = {xi: [] for xi in x_axis}

        evals_scores = exp.get_eval(self.reward_key)
        it_idx = 0
        for instance_chunk in tqdm(
            chunked(exp.instances(lazy_iterable=True), self.chunk_size)
        ):

            evals_scores_chunk = evals_scores[it_idx : it_idx + len(instance_chunk)]
            it_idx += len(instance_chunk)

            is_quest = "quest" in exp.meta["variant"]

            instances_and_scores = [
                join_instances_and_repeat_accepts_with_scores(
                    [ich], evals_scores=[ech], is_quest=is_quest
                )
                for ich, ech in zip(instance_chunk, evals_scores_chunk)
            ]

            for i in x_axis:

                results = self.pmap(
                    partial(
                        is_weighted_mode_correct,  # is_weighted_mode_correct
                        n=i * 1,
                        extract_func=self.extract,
                        beta=self.beta,
                    ),
                    instances_and_scores,
                )

                accs, estimates = zip(*results)

                accs_full[i].extend(accs)

        results = [
            {
                "axis": x_axis,
                "scores": [float(np.mean(accs_full[i])) for i in x_axis],
                "accs": [accs_full[i] for i in x_axis],
            }
        ]
        return results


class MBRPick(Evalutor):

    def __init__(
        self,
        gaps=2,
        extract="lastnumber",
        n=None,
        metric="rouge",
        **kwargs,
    ):
        super().__init__(f"{metric}-mbr" if n is None else f"{metric}-mbr-" + str(n))
        self.gaps = gaps

        self.metric = metric
        self.n = n
        print(f"Filename:", self.eval_name)

        self.extract_key = extract

    def eval(self, exp: Exp):

        if not self.n:
            n = exp.meta["n"]
        else:
            n = self.n

        pairwise_pick_results = mbr_pick_progression(
            exp,
            compute_in_parallel=True,
            k=self.gaps,
            n=n,
            metric=self.metric,
        )

        gt = exp.get_eval(self.extract_key)[:n]

        scores = {}
        for ax, preds in pairwise_pick_results["preds"].items():
            pred_scores = [gt[j]["scores"][pred] for j, pred in enumerate(preds)]
            scores[ax] = float(np.mean(pred_scores))

        output = {
            "axis": pairwise_pick_results["axis"],
            "scores": [scores[ax] for ax in pairwise_pick_results["axis"]],
        }

        print(output)

        return [output]


class WMBRPick(Evalutor):

    def __init__(
        self,
        key,
        gaps=2,
        extract="lastnumber",
        n=None,
        metric="rouge",
        **kwargs,
    ):
        super().__init__(f"{metric}-wmbr" if n is None else f"{metric}-wmbr-" + str(n))
        self.gaps = gaps
        self.metric = metric
        self.n = n
        self.reward_key = key
        print(f"Filename:", self.eval_name)

        self.extract_key = extract

    def eval(self, exp: Exp):

        if not self.n:
            n = exp.meta["n"]
        else:
            n = self.n

        pairwise_pick_results = weighted_mbr_pick_progression(
            exp,
            reward_key=self.reward_key,
            compute_in_parallel=True,
            k=self.gaps,
            n=n,
            metric=self.metric,
        )

        gt = exp.get_eval(self.extract_key)[:n]

        scores = {}
        for ax, preds in pairwise_pick_results["preds"].items():
            pred_scores = [gt[j]["scores"][pred] for j, pred in enumerate(preds)]
            scores[ax] = float(np.mean(pred_scores))

        output = {
            "axis": pairwise_pick_results["axis"],
            "scores": [scores[ax] for ax in pairwise_pick_results["axis"]],
        }

        print(output)

        return [output]
