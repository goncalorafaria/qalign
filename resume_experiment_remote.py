## qalign
from qalign.utils.experiment import run_experiment_remote

## expkit
from expkit.storage import DiskStorage
from expkit import Exp


# 280bd4d8-9e11-48d4-812b-fd2c5666684d
def main(
    experiment_name="",
    save_path: str = "llama3.2-outputs/",
):

    storage = DiskStorage(save_path, "rw")

    if storage.exists(experiment_name):
        experiment = Exp.load(storage=storage, name=experiment_name)

        run_experiment_remote(
            experiment=experiment,
        )

    else:
        raise ValueError(f"Experiment {experiment_name} does not exist.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
