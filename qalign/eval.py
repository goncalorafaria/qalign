from typing import *

from qalign.reward import RemoteReward

## expkit
from expkit.setup import ExpSetup
from expkit.storage import DiskStorage

## qalign
from qalign.utils.eval import *


def main(
    base_dir="remote-outputs-llama/",
    reward_model_path="lastnumber",
    model_url="http://localhost:8080",
    query_args={},
    n=None,
):

    print("Query Args:", query_args)
    setup = ExpSetup(storage=DiskStorage(base_dir=base_dir, mode="rw"))
    # print("Exp:", setup.query(query_args))

    setup = setup.query(query_args).filter(lambda x: x.has_data())

    print("That match the query:\n", setup)

    if len(setup.experiments) == 0:
        raise FileNotFoundError("The experiment has no data!")

    if  reward_model_path == "lastnumber":
        ps_eval = ExactLastNumberEval()

    elif reward_model_path == "lastmath":
        ps_eval = ExactMATHEval()

    elif reward_model_path == "lastoption":
        ps_eval = ExactQAEval()

    elif reward_model_path == "ifeval":
        ps_eval = IFEval()

    else:

        
        reward = RemoteReward(
            model_path=reward_model_path,
            server_url=model_url,
        )
        
        ps_eval = RewardEval(
            reward=reward,
            n=n,
            chunk_size=256,
        )
        
    print("That haven't done the eval:", setup)

    def func(experiment):

        try:
            return (
                ps_eval(experiment)
                # if not experiment.has_eval(ps_eval.eval_name)
                # else experiment
            )
        except FileNotFoundError:
            return experiment

        except Exception as e:
            raise e
            # return experiment

    setup = setup.map(func)

    # new_setup.save()


if __name__ == "__main__":

    import fire

    fire.Fire(main)
