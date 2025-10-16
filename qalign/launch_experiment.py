from qalign.utils.experiment import create_experiment


def main(
    variant="quest-rlhf",
    beta: float = 1.0,
    steps: int = 4096,
    temperature: float = 1.0,
    n: int = 64,
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    reward_model_path: str = "/gscratch/ark/graf/quest-rlhf/qflow/rm/artifacts/llama3/8b8b/mathcotfix/full/reward",  # "/gscratch/ark/graf/LLaMA-Factory/saves/llama3/8b/full/reward/",
    reward_model_batch_size: int = 4,
    save_path: str = "remote-outputs-llama/",
    gpu_memory_utilization: float = 0.8,
    max_new_tokens: int = 800,
    max_prompt_length: int = 1200,
    dataset_path="HuggingFaceH4/MATH-500",  # "apple/GSM-Symbolic-p1",
    batch_size=64,
    stop_tokens=[],
    prompt_template="{prompt}",
    num_chains: int = 1,
    use_few_shot: bool = False,
    format: str = "chat",
    split: str = "test",
    remote: bool = True,
    reward_type="value",
):

    additional_meta = {
        "remote": remote,
    }

    if variant == "quest-rlhf":

        # Create experiment with additional meta data
        additional_meta = {
            "beta": beta,
            "reward_model_path": reward_model_path,
            "reward_type": reward_type,
            # "i": start_index,
            "num_chains": num_chains,
            **additional_meta,
        }

    experiment = create_experiment(
        save_path=save_path,
        variant=variant,
        model_path=model_path,
        dataset_path=dataset_path,
        n=n,
        temperature=temperature,
        steps=steps,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_prompt_length,
        stop_tokens=stop_tokens,
        prompt_template=prompt_template,
        additional_meta=additional_meta,
        batch_size=batch_size,
        use_few_shot=use_few_shot,
        format=format,
        split=split,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
