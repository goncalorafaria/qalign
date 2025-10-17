from itertools import islice
from typing import List, Dict, Any
from collections import OrderedDict


def unflatten_list(flat_data, counts):

    unflattened_translations = []
    start = 0
    for count in counts:
        end = start + count
        unflattened_translations.append(flat_data[start:end])
        start = end

    return unflattened_translations



def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def chunked(iterator, size):
    iterator = iter(iterator)
    return iter(lambda: list(islice(iterator, size)), [])


class OrderedSet:
    def __init__(self, elements=[]):
        self.elements = OrderedDict()

        for e in elements:
            self.add(e)

    def add(self, item):
        self.elements[item] = None

    def remove(self, item):
        del self.elements[item]

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, item):
        return item in self.elements

    def __len__(self):
        return len(self.elements)



def get_unique_mapping(si):
    vocab = {s: i for i, s in enumerate(OrderedSet(si))}
    tokens = [vocab[s] for s in si]
    sorted_vocab = [
        k
        for k, v in sorted(
            vocab.items(),
            key=lambda item: item[1],
        )
    ]

    return tokens, sorted_vocab


def join_accepted_values(accept, proposal, state):
    return [p if a else s for a, p, s in zip(accept, proposal, state)]


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


def invert_unique_mapping(tokens, rsi):
    return [rsi[t] for t in tokens]


def split_into_groups(lst, batch_size=8):
    if not lst:
        return []

    # Determine number of groups (max 8)
    n = min(batch_size, len(lst))

    # Calculate size of each group
    base_size = len(lst) // n
    remainder = len(lst) % n

    result = []
    start = 0

    for i in range(n):
        # Add one extra element to early groups if there's remainder
        group_size = base_size + (1 if i < remainder else 0)
        end = start + group_size
        result.append(lst[start:end])
        start = end

    return result
