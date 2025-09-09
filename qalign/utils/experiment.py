from typing import Optional, Dict, Any, List
from datetime import datetime
from tqdm import tqdm
import os
import copy

## expkit
from expkit.exp import Exp
from expkit.storage import DiskStorage

## qalign
from qalign.utils.data import FlexiblePromptTemplate
from qalign.utils.data import get_data_iterable
from qalign.utils.list import chunked


from qalign.reward import RemoteReward
from qalign.model import RemoteVLLM
from qalign.base import QAlign


def create_experiment(
    save_path: str,
    variant: str,
    model_path: str,
    dataset_path: str,
    n: int,
    temperature: float,
    steps: int,
    max_new_tokens: int = 100,
    max_prompt_length: int = 600,
    batch_size: int = 64,
    split: str = "test",
    prompt_template: str = "{prompt}",
    stop_tokens: Optional[List[str]] = None,
    additional_meta: Optional[Dict[str, Any]] = None,
    format: str = "chat",  # either "chat" or "prompt"
    use_few_shot: bool = False,
) -> Exp:
    """
    Creates a standardized experiment with common metadata.
    """
    if stop_tokens is None:
        stop_tokens = []

    meta = {
        "steps": steps,
        "temperature": temperature,
        "n": n,
        "model_path": model_path,
        "variant": variant,
        "stop_tokens": stop_tokens,
        "max_new_tokens": max_new_tokens,
        "max_prompt_length": max_prompt_length,
        "at": datetime.now().isoformat(),
        "dataset": dataset_path,
        "split": split,
        "prompt_template": prompt_template,
        "batch_size": batch_size,
        "format": format,
        "use_few_shot": use_few_shot,
    }

    if additional_meta:
        meta.update(additional_meta)

    return Exp(
        storage=DiskStorage(save_path, "rw"),
        meta=meta,
    )


def create_extension_experiment(storage, experiment, new_steps=1024):
    samples = [
        {
            "input": (
                data["input"]["input"] if "input" in data["input"] else data["input"]
            ),
            "completion": data["outputs"][-1]["text"],
            "reward": float(data["outputs"][-1]["reward"]),
        }
        for data in experiment.instances(lazy_iterable=True)
    ]

    meta = copy.deepcopy(experiment.meta)

    meta["steps"] = new_steps
    meta["bootstrap"] = samples
    meta["link"] = experiment.get_name()

    new_exp = Exp(
        storage=storage,
        name=experiment.get_name() + "-extension",
        meta=meta,
    )

    return new_exp



def process_batch_outputs(
    chain_outputs: Any, batch_size: int
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
                for s in chain_outputs.state_path
            ]
        )
    return outputs


def get_batched_data(
    model,
    dataset_path: str,
    split: str,
    n: int,
    batch_size: int,
    prompt_template: str,
    start_index: int = 0,
    num_chains: int = 1,
    completed: int = 0,
    format="chat",
    use_few_shot=False,
) -> List[Any]:
    """
    Gets batched data from a dataset using standard configurations.
    """
    data_iterable = get_data_iterable(
        model_path=model.model_path,
        dataset_path=dataset_path,
        split=split,
        n=start_index + n,
        tokenizer=model.tokenizer,
        prompt_template=FlexiblePromptTemplate(prompt_template),
        format=format,
        use_few_shot=use_few_shot,
    )

    if start_index > 0:
        data_iterable = data_iterable[start_index : start_index + n]

    if num_chains > 1:
        data_iterable = [x for x in data_iterable for _ in range(num_chains)]

    data_iterable = data_iterable[completed:]

    batches = []
    for i in range(0, len(data_iterable), batch_size):
        batches.append(data_iterable[i : i + batch_size])

    return batches


def calculate_reward_scores(
    experiment: Exp,
    reward_key: Optional[str] = None,
) -> List[Dict[str, List[float]]]:
    """
    Calculates reward scores for experiment instances.
    """
    beta = experiment.meta.get("beta", 1.0)

    if not reward_key and "reward_model_path" in experiment.meta:
        reward_key = "crm:" + experiment.meta["reward_model_path"].split(".")[
            0
        ].replace("/", "-")

    return [
        {"scores": [float(o["reward"]) * beta for o in i["outputs"]]}
        for i in experiment.instances(lazy_iterable=True)
    ]


def run_ancestral(experiment, model, steps, data_batches):

    steps = experiment.meta["steps"]

    
    # Process each batch
    for data_batch in tqdm(data_batches):
        
        
        prompts = [ model.tokenizer.apply_chat_template(
            d["chat_template_prompt"],
            tokenize=False,
            add_generation_prompt=True,
        ) for d in data_batch ]
        
        
        completions_txt = model.ancestral(prompts, n=steps)
        outputs = [
            [{"text": state_t} for state_t in instance_txt]
            for instance_txt in completions_txt
        ]

        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )


def run_quest(
    experiment,
    model,
    steps,
    data_batches,
    reward_model_batch_size=16,
    reward_url="http://localhost:8080",
):

    

    reward = RemoteReward(
        server_url=reward_url,
        model_path=experiment.meta["reward_model_path"],
        batch_size=reward_model_batch_size,
    )
    
    # Process each batch
    for data_batch in data_batches:
 
        chain = QAlign(
            model=model,
            reward=reward,
            beta=experiment.meta["beta"],
        )

        chain_outputs = chain.run_pipelined(
            prompts=[data["prompt"] for data in data_batch],
            steps=steps,
            use_tqdm=True,
            workers=4,
        )

        outputs = chain_outputs.state_path
        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )

    # Calculate and add reward scores
    scores = calculate_reward_scores(experiment)
    experiment.add_eval(reward.get_name(), scores)

    return experiment


def run_quest_bootstrap(
    experiment,
    model,
    steps,
    reward_model_batch_size=16,
    reward_url="http://localhost:8080",
):


    reward = RemoteReward(
        server_url=reward_url,
        model_path=experiment.meta["reward_model_path"],
        batch_size=reward_model_batch_size,
    )
    
    for data_batch in chunked(
        experiment.meta["bootstrap"], experiment.get("batch_size")
    ):

        chain = QAlign(
            model=model,
            reward=reward,
            beta=experiment.meta["beta"],
        )

        chain_outputs = chain.run_pipelined(
            prompts=[data["prompt"] for data in data_batch],
            steps=steps,
            use_tqdm=True,
            warm_start=[
                {"completion": data["completion"], "reward": data["reward"]}
                for data in data_batch
            ],
            workers=4,
        )

        outputs = chain_outputs.state_path

        experiment.add_instances(
            inputs=data_batch,
            outputs=outputs,
        )

    # Calculate and add reward scores
    scores = calculate_reward_scores(experiment)
    experiment.add_eval(reward.get_name(), scores)

    return experiment


#  Create model
def run_experiment(
    experiment,
    model_url="http://localhost:8080",
    reward_url="http://localhost:8080",
    reward_model_batch_size=16,
):

    # Create model
    model =  RemoteVLLM(
        server_url=model_url,
        model_path=experiment.meta["model_path"],
        max_prompt_length=experiment.meta["max_prompt_length"],
        max_new_tokens=experiment.meta["max_new_tokens"],
    )

    completed = len(experiment.instances())

    # Get batched data with start index
    if "bootstrap" in experiment.meta:

        if "bootstrap" in experiment.meta:
            run_quest_bootstrap(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                reward_model_batch_size=reward_model_batch_size,
                reward_url=reward_url,
            )

    else:
            
        # Get batched data with start index
        data_batches = get_batched_data(
            model=model,
            dataset_path=experiment.meta["dataset"],
            split=experiment.meta["split"],
            n=experiment.meta["n"],
            batch_size=experiment.meta.get("batch_size", 64),
            prompt_template=experiment.meta["prompt_template"],
            start_index=experiment.meta.get("i", 0),
            num_chains=experiment.meta.get("num_chains", 1),
            completed=completed,
            format=experiment.meta.get("format", "chat"),
            use_few_shot=experiment.meta.get("use_few_shot", False),
        )

        if experiment.meta["variant"] == "ancestral":
            run_ancestral(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                data_batches=data_batches,
            )
        elif experiment.meta["variant"] == "quest-rlhf":
            run_quest(
                experiment=experiment,
                model=model,
                steps=experiment.meta["steps"],
                data_batches=data_batches,
                reward_model_batch_size=reward_model_batch_size,
                reward_url=reward_url,
            )


run_experiment_remote=run_experiment
