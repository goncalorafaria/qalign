from qalign.utils.experiment import create_extension_experiment, run_experiment_remote
from expkit.storage import DiskStorage
from expkit import Exp


def main(
    base_experiment_name: str,
    save_path: str = "outputs-quest/",
    new_steps: int = 64,
):
    """
    Creates and runs a bootstrap experiment based on an existing experiment.
    
    Args:
        base_experiment_name: Name of the base experiment to extend
        save_path: Path where experiments are stored
        new_steps: Number of steps for the bootstrap experiment
        model_url: URL of the model server
        reward_url: URL of the reward server
        num_workers: Number of workers for parallel processing
        max_concurrent_requests: Maximum concurrent requests to the model server
        batch_size: Batch size for processing
    """
    storage = DiskStorage(save_path, "rw")

    if not storage.exists(base_experiment_name):
        raise ValueError(f"Experiment {base_experiment_name} does not exist in {save_path}.")

    # Load the base experiment
    base_experiment = Exp.load(storage=storage, name=base_experiment_name)

    # Create extension experiment with bootstrap data
    extension_experiment = create_extension_experiment(
        storage=storage,
        experiment=base_experiment,
        new_steps=new_steps,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

