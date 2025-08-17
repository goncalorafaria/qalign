from fastapi import HTTPException
from pydantic import BaseModel
import torch
from typing import List, Optional, Dict
import uvicorn
from fire import Fire
import asyncio
from queue import Queue
import uuid
import time
from threading import Thread
import os
import random
import numpy as np
import gc
import socket


## literegistry
from literegistry import ServiceAPI

## quest
from qalign.reward import ClassificationRewardModel, ValueHead
#from quest.reward.mt import QEModel
#from quest.utils.logger import fix_loggers

# Add this at the top of your script, before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global instances
MODEL = None
TASK_QUEUE = Queue()
RESULTS: Dict[str, dict] = {}
TASK_THREAD = None


class PredictionRequest(BaseModel):
    input: List[List[Dict[str, str]]]
    use_tqdm: bool = False


class TaskResponse(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    status: str
    rewards: Optional[List[float]] = None
    error: Optional[str] = None
    extra: Optional[List[str]] = None


def clear_gpu_memory():
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        gc.collect()


def process_queue():
    while True:
        try:
            if not TASK_QUEUE.empty():
                task_id, request = TASK_QUEUE.get()
                try:


                    # Process the request
                    rewards = MODEL.evaluate(
                        conversations=request.input,
                    )

                    RESULTS[task_id] = {"status": "completed", "rewards": rewards}

                    clear_gpu_memory()

                except Exception as e:
                    print(f"Error processing task {task_id}: {str(e)}")
                    RESULTS[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "extra": [len(t) for t in request.texts],
                    }
                    print("falty packet? :" + str(request.texts))
                    import traceback

                    tb = traceback.format_exc()
                    print(f"Error processing task {task_id}:\n{tb}")
                finally:
                    TASK_QUEUE.task_done()
            else:
                time.sleep(0.1)  # Small delay when queue is empty
        except Exception as e:
            print(f"Error in queue processing: {str(e)}")
            time.sleep(1)  # Delay on error


def create_server(
    reward_type: str = "value",
    reward_model_path: str = "/gscratch/ark/graf/quest-rlhf/qflow/rm/artifacts/llama3/8b8b/mathcot/full/reward",
    reward_model_batch_size: int = 8,
    reward_device: int = 0,
    reward_device_count: int = 1,
    host: str = "0.0.0.0",
    request_timeout: int = 60 * 2,
    poll_interval: int = 1,
    registry_path="/gscratch/ark/graf/registry",
):

    port = random.randint(8000, 10000)
    metadata = {
        "reward_type": reward_type,
        "device": reward_device,
        "device_count": reward_device_count,
        "model_path": reward_model_path,
        "route": "/evaluate",
        "args": ["context", "texts"],
    }

    app = ServiceAPI(
        title="Reward Model Server",
        hostname=f"{socket.gethostname()}.hyak.local",
        port=port,
        metadata=metadata,
        registry_path=registry_path,
    )

    @app.on_event("startup")
    async def startup():
        global MODEL, TASK_THREAD
        try:
            if reward_type == "contextual":
                MODEL = ClassificationRewardModel(
                    model_path=reward_model_path,
                    batch_size=reward_model_batch_size,
                    device=reward_device,
                    device_count=reward_device_count,
                )
            elif reward_type == "value":
                MODEL = ValueHead(
                    model_path=reward_model_path,
                    batch_size=reward_model_batch_size,
                    device=reward_device,
                    device_count=reward_device_count,
                )
            
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

            # Start queue processing thread
            TASK_THREAD = Thread(target=process_queue, daemon=True)
            TASK_THREAD.start()

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )

    @app.post("/classify", response_model=TaskStatus)
    async def evaluate(request: PredictionRequest):
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        task_id = str(uuid.uuid4())
        RESULTS[task_id] = {"status": "pending"}
        TASK_QUEUE.put((task_id, request))

        print(f"Task {task_id} added to queue. batchsize: {len(request.texts)}")
        # Poll for the result.
        timeout = request_timeout  # maximum time (in seconds) to wait for a result
        elapsed = 0.0

        while elapsed < timeout:
            result = RESULTS.get(task_id)
            if result and result["status"] in ["completed", "failed"]:
                return TaskStatus(**result)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout reached: return pending status and task_id so the client can poll later.
        raise HTTPException(status_code=504, detail="Task processing took too long")

    @app.get("/queue_size")
    async def get_queue_size():
        return {"queue_size": TASK_QUEUE.qsize()}

    @app.get("/health")
    async def health_check():
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if not TASK_THREAD or not TASK_THREAD.is_alive():
            raise HTTPException(
                status_code=503, detail="Task processing thread not running"
            )
        return {
            "status": "healthy",
            "model_type": reward_type,
            "device": f"cuda:{reward_device}" if torch.cuda.is_available() else "cpu",
            "queue_size": TASK_QUEUE.qsize(),
        }

    # Fix loggers
    #fix_loggers(name="transformers")

    # Run the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    Fire(create_server)
