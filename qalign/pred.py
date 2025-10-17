from typing import *
from multiprocessing import Pool


## expkit
from expkit.setup import ExpSetup
from expkit.storage import DiskStorage

## qalign
from qalign.utils.pred import *
from qalign.utils.eval import *


def main(
    base_dir="remote-outputs/",  # "llama3.2-outputs/",  # "tqa-outputs/",
    strategy="voting",
    key="crm:allenai-Llama-3",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-tulu-8b8b-gsm8k-full-reward-",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-tulu-8b8b-math-full-reward-",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-tulu-8b8b-gsm8k-full-reward-",  # "crm:allenai-Llama-3",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-llama3-8b8b-mathcot-full-reward",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-llama3-8b8b-math-full-reward-",  # "vh:-gscratch-ark-graf-quest-rlhf-qflow-rm-artifacts-llama3-8b8b-math-full-reward-",  # "vh:-gscratch-ark-graf-LLaMA-Factory-saves-llama3-8b8b-full-reward-",  # "vh:-gscratch-ark-graf-LLaMA-Factory-saves-llama3-3b-full-reward-",
    query_args={
        # "steps": 4096,
        # "split": "test",
        # "split": "validation",
        "temperature": 1.0,
        # "beta": 0.5,
        # "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset": "HuggingFaceH4/MATH-500",  # "HuggingFaceH4/MATH-500",  # "openai/gsm8k",  # HuggingFaceH4/MATH-500
        # "dataset": "lighteval/MATH",
        # "model_path": "allenai/Llama-3.1-Tulu-3-8B-SFT",
        # "reward_model_path": "/gscratch/ark/graf/LLaMA-Factory/saves/llama3/8b1b/full/reward/",
        # "reward_model_path": "allenai/Llama-3.1-Tulu-3-8B-RM",
        # "variant": "ancestral",
        # "n": 128,
    },
    beta: float = 1.0,
    c: float = 1.0,
    extract="lastmath",
    trials=512,
    r=32,
    gaps=64,
    exp_rate=False,
    n=None,
):

    setup = (
        ExpSetup(storage=DiskStorage(base_dir=base_dir, mode="rw"))
        .query(query_args)
        .filter(lambda x: x.has_data())
    )

    print(setup)
    # setup.experiments[0].evals()

    print(
        len(setup.experiments),
    )
    if len(setup.experiments) == 0:
        raise FileNotFoundError("The configuration has no data!")

    # strategy, reward_model_path = strategy.split("-")

    with Pool() as p:
        pick = get_strategy(
            strategy,
            key=key,
            p=p,
            beta=beta,
            c=c,
            extract=extract,
            trials=trials,
            r=r,
            gaps=gaps,
            exp_rate=exp_rate,
            n=n,
        )  # msft , nvdia

        # setup = setup.filter(lambda x: not x.has_eval(pick.eval_name))

        def func(experiment):

            try:
                # if len(experiment.instances()) == experiment.meta["n"]:

                # print(len(experiment.instances()))
                return (
                    pick(experiment)
                    # if not experiment.has_eval(pick.eval_name)
                    # else experiment
                )
                # else:
                #    return experiment

            except FileNotFoundError:
                print(experiment.name)
                return experiment

            except Exception as e:
                print(experiment.name)
                print(e)
                raise e
                return experiment

        setup = setup.map(func)

    # new_setup.save()


if __name__ == "__main__":

    import fire

    fire.Fire(main)
