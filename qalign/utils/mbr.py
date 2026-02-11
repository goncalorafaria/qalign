import numpy as np
from collections import Counter

from typing import *

from multiprocessing import Pool
import numpy as np
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from scipy.sparse import csr_matrix


## qalign
from qalign.utils.math import generate_axis
from qalign.utils.list import (
    chunked,
)


class UCS:

    def __init__(
        self,
        process_num=None,
        vocab_size=1000,
    ):

        self.process_num = os.cpu_count() if process_num is None else process_num
        self.vocab_size = vocab_size
        self.func = ucs_sparse

    def _process_chunk(self, chunk):
        """
        Process a chunk of data - this is needed for multiprocessing.

        Args:
            chunk: A subset of samples to process

        Returns:
            UCS matrix for the chunk
        """
        return self.func(chunk, vocab_size=self.vocab_size)

    def get_score(self, samples, compute_in_parallel=True):

        if compute_in_parallel:
            with Pool(self.process_num) as executor:
                results = list(
                    tqdm(
                        executor.map(
                            self._process_chunk,
                            samples,
                        ),
                        total=len(samples),
                        desc="Processing chunks",
                    )
                )

                mat = np.array(results)
        else:
            mat = np.array(
                [
                    self.func(generations, vocab_size=self.vocab_size)
                    for generations in samples
                ]
            )

        return mat


class FastROUGE(UCS):
    def __init__(
        self,
        process_num=None,
        vocab_size=1000,
    ):

        self.process_num = os.cpu_count() if process_num is None else process_num
        self.vocab_size = vocab_size
        self.func = rouge1_sparse


def join_instances_accepts(i, is_quest=True):

    if is_quest:
        accepted_outputs = []
        accepted_index = []
        last_accepted = None
        last_accept_index = None
        for j, output in enumerate(i["outputs"]):
            if output["accept"]:
                accepted_outputs.append(output)
                accepted_index.append(j)
                last_accepted = output
                last_accept_index = j
            elif last_accepted is not None:
                accepted_outputs.append(last_accepted)
                accepted_index.append(last_accept_index)

    else:
        accepted_outputs = i["outputs"]
        accepted_index = list(range(len(i["outputs"])))

    # print(len(outputs))
    return {"input": i["input"], "outputs": accepted_outputs, "index": accepted_index}


def get_mats(completions, compute_in_parallel=True, metric="bleu", vocab_size=1000):

    if "ucs" in metric:
        s = UCS(vocab_size=vocab_size)
    elif "rouge" in metric:
        s = FastROUGE(vocab_size=vocab_size)  # PairwiseRouge(metric=metric)
    else:
        raise ValueError("metric must be bleu or rouge")

    mats = s.get_score(completions, compute_in_parallel=compute_in_parallel)

    return mats


def ucs_sparse(generations, vocab_size):
    """
    Compute Unigram Consistency Score (UCS) matrix for lists of integers using
    a vectorized approach for better efficiency.

    Args:
        int_lists: List of integer lists where each list represents a sequence of elements
        vocab_size: Size of the vocabulary |V| to use in the UCS calculation

    Returns:
        UCS matrix where UCS[i, j] represents the consistency score between
        lists i and j
    """
    n = len(generations)

    # Check if provided vocab_size is valid
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be greater than 0")

    # Create a binary matrix of shape (n, vocab_size) where binary_matrix[i, j] = 1
    # if element j is in list i, 0 otherwise

    # Method 1: Using sparse matrices (more efficient for large vocab_size)
    rows = []
    cols = []
    data = []

    for i, int_list in enumerate(generations):
        for elem in set(int_list):  # Use set to count each unique element once
            if 0 <= elem < vocab_size:  # Ensure the element is in range
                rows.append(i)
                cols.append(elem)
                data.append(1)

    # Create sparse matrix
    binary_matrix = csr_matrix((data, (rows, cols)), shape=(n, vocab_size))

    # Compute UCS matrix using matrix multiplication
    # (binary_matrix @ binary_matrix.T) gives the dot product of each pair of rows
    dot_products = binary_matrix @ binary_matrix.T

    # Convert to dense array and divide by vocab_size
    ucs_matrix = dot_products.toarray() / vocab_size

    return ucs_matrix


def rouge1_sparse(generations, vocab_size):
    """
    Compute Unigram Consistency Score (UCS) matrix for lists of integers using
    a vectorized approach for better efficiency.

    Args:
        int_lists: List of integer lists where each list represents a sequence of elements
        vocab_size: Size of the vocabulary |V| to use in the UCS calculation

    Returns:
        UCS matrix where UCS[i, j] represents the consistency score between
        lists i and j
    """
    n = len(generations)

    # Check if provided vocab_size is valid
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be greater than 0")

    # Create a binary matrix of shape (n, vocab_size) where binary_matrix[i, j] = 1
    # if element j is in list i, 0 otherwise

    # Method 1: Using sparse matrices (more efficient for large vocab_size)
    rows = []
    cols = []
    data = []
    # Store total word counts for each generation (for normalization)
    word_counts = np.zeros(n)

    for i, int_list in enumerate(generations):
        # Count total words in this generation (including duplicates)
        word_counts[i] = len(int_list)

        # Count frequency of each word in this generation
        # word_freq = {}
        # for word in int_list:
        #    if 0 <= word < vocab_size:  # Ensure word is in valid range
        #        word_freq[word] = word_freq.get(word, 0) + 1

        word_freq = Counter(int_list)

        # Add to sparse matrix data
        for word, freq in word_freq.items():
            if 0 <= word < vocab_size:
                rows.append(i)
                cols.append(word)
                data.append(freq)  # S

    count_matrix = csr_matrix((data, (rows, cols)), shape=(n, vocab_size))

    overlap_matrix = np.zeros((n, n), dtype=np.float32)

    # Convert to binary matrix first (1 where count > 0)
    binary_matrix = count_matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)

    # Get common token indicators using matrix multiplication
    common_tokens = binary_matrix @ binary_matrix.T

    # For each pair of rows, calculate the overlap
    for i in range(n):
        row_i = count_matrix.getrow(i)

        for j in range(i, n):
            # Only process if there are any common tokens
            if common_tokens[i, j] > 0:
                row_j = count_matrix.getrow(j)

                # Get common indices
                i_indices = row_i.indices
                i_data = row_i.data
                j_indices = row_j.indices
                j_data = row_j.data

                # Find the intersection of indices
                # This is more efficient than multiplying sparse matrices
                i_dict = dict(zip(i_indices, i_data))
                j_dict = dict(zip(j_indices, j_data))

                common_indices = set(i_indices).intersection(set(j_indices))

                # Calculate the sum of minimums
                overlap = sum(min(i_dict[idx], j_dict[idx]) for idx in common_indices)

                overlap_matrix[i, j] = overlap
                if i != j:
                    overlap_matrix[j, i] = overlap

    # Calculate ROUGE-1 recall (overlap / words in reference)
    # We can use broadcasting to divide by word_counts
    recall_matrix = np.zeros((n, n))
    nonzero_counts = word_counts > 0
    if np.any(nonzero_counts):
        recall_matrix[nonzero_counts, :] = (
            overlap_matrix[nonzero_counts, :] / word_counts[nonzero_counts, np.newaxis]
        )

    # Calculate ROUGE-1 precision (overlap / words in candidate)
    precision_matrix = np.zeros((n, n))
    if np.any(nonzero_counts):
        precision_matrix[:, nonzero_counts] = (
            overlap_matrix[:, nonzero_counts] / word_counts[np.newaxis, nonzero_counts]
        )

    # Calculate ROUGE-1 F1 score
    f1_matrix = np.zeros((n, n))
    nonzero = (precision_matrix + recall_matrix) > 0
    f1_matrix[nonzero] = (
        2
        * (precision_matrix[nonzero] * recall_matrix[nonzero])
        / (precision_matrix[nonzero] + recall_matrix[nonzero])
    )

    return {
        "recall": recall_matrix,
        "precision": precision_matrix,
        "f1": f1_matrix,
        "overlap": overlap_matrix,
        "word_counts": word_counts,
    }["f1"]


def mbr_mat_progression(
    exp,
    compute_in_parallel=True,
    k=1,
    n=None,
    max_steps=None,
    metric="bleu",
):

    if max_steps is None:
        max_steps = exp.get("steps")

    # total_instances = exp.instances(lazy_iterable=True)

    # if n is not None:
    #    total_instances = total_instances[:n]

    # if "quest" not in exp.get("variant"):
    tokenizer = AutoTokenizer.from_pretrained(exp.get("model_path"))

    mats = []

    repeat_inds = []

    for instances in tqdm(chunked(exp.instances(lazy_iterable=True), 32)):
        # instances = [instance]

        instances = [
            join_instances_accepts(i, is_quest="quest" in exp.get("variant"))
            for i in instances
        ]

        repeat_inds.extend([i["index"] for i in instances])

        if "quest" not in exp.get("variant"):

            texts = [
                o["text"]
                for instance in instances
                for o in instance["outputs"][:max_steps]  # Already uniform length
            ]

            # Batch tokenize everything at once
            batch_ids = tokenizer(texts)["input_ids"]

            # Reshape into [num_instances, max_steps, ...] using fixed chunk size
            completions = [
                batch_ids[i : i + max_steps]
                for i in range(0, len(batch_ids), max_steps)
            ]

        else:
            completions = [
                [o["completion"] for o in instance["outputs"][:max_steps]]
                for instance in instances
            ]

        mat = get_mats(
            completions,
            compute_in_parallel=compute_in_parallel,
            metric=metric,
            vocab_size=tokenizer.vocab_size,
        )

        mats.append(mat)

        del instances
        del completions

    mat = np.concatenate(mats, axis=0)

    return mat, repeat_inds


def weighted_mbr_pick_progression(
    exp,
    reward_key,
    compute_in_parallel=True,
    k=1,
    n=None,
    max_steps=None,
    metric="bleu",
    beta=1.0,
):

    if max_steps is None:
        max_steps = exp.get("steps")

    mat, repeat_inds = mbr_mat_progression(
        exp,
        compute_in_parallel=compute_in_parallel,
        k=k,
        n=n,
        max_steps=max_steps,
        metric=metric,
    )

    rewards = exp.get_eval(reward_key)

    repeat_rewards = np.array(
        [
            [r["scores"][i] for i in inds][:max_steps]
            for inds, r in zip(repeat_inds, rewards)
        ]
    )
    
    #softmax the rewards
    from scipy.special import softmax
    repeat_rewards = softmax(repeat_rewards/beta, axis=1)

    repeat_rewards = np.expand_dims(repeat_rewards, axis=1)

    mat *= repeat_rewards

    axis = generate_axis(max_steps + 1, k)

    return pick_mat(mat, axis, repeat_inds)


def mbr_pick_progression(
    exp,
    compute_in_parallel=True,
    k=1,
    n=None,
    max_steps=None,
    metric="bleu",
):

    if max_steps is None:
        max_steps = exp.get("steps")

    mat, repeat_inds = mbr_mat_progression(
        exp,
        compute_in_parallel=compute_in_parallel,
        k=k,
        n=n,
        max_steps=max_steps,
        metric=metric,
    )

    axis = generate_axis(max_steps + 1, k)

    # 1,2,4,8,16,32,64,96,128,160,192,224,256

    return pick_mat(mat, axis, repeat_inds)


def pick_mat(
    mat,
    axis,
    repeat_inds,
):

    values = []

    preds = {}
    for ni in axis:
        pick_batch = np.argmax(mat[:, :ni, :ni].mean(axis=-1), axis=-1)
        # acc = []

        preds[ni] = []
        for i, pick in enumerate(
            pick_batch,
        ):
            # true_answer = extract_func(instance["input"]["answer"])
            pred_index = repeat_inds[i][pick]
            preds[ni].append(pred_index)
            # acc.append(correct)

    return {"preds": preds, "axis": axis}
