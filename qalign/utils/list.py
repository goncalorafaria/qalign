from itertools import islice
from typing import List, Dict, Any


def unflatten_list(flat_data, counts):

    unflattened_translations = []
    start = 0
    for count in counts:
        end = start + count
        unflattened_translations.append(flat_data[start:end])
        start = end

    return unflattened_translations



def chunked(iterator, size):
    iterator = iter(iterator)
    return iter(lambda: list(islice(iterator, size)), [])



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
