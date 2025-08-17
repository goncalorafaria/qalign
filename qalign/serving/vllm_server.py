import subprocess
import requests
import time
import socket
import fire
import threading
import asyncio
from typing import Tuple
import re
import math
import random


from literegistry import ServerRegistry, FileSystemKVStore


def parse_vllm_requests(metrics_text: str) -> Tuple[float, float]:
    """
    Parse VLLM metrics text and extract running and waiting requests.

    Args:
        metrics_text (str): Raw prometheus-style metrics text

    Returns:
        Tuple[float, float]: A tuple containing (running_requests, waiting_requests)

    Example:
        running, waiting = parse_vllm_requests(metrics_text)
        total_requests = running + waiting
    """
    # Pattern to match metric lines with their values
    pattern = r"vllm:num_requests_(\w+){.*?} (-?\d+\.?\d*)"

    # Find all matches in the text
    matches = re.finditer(pattern, metrics_text)

    # Initialize values
    metrics = {"running": 0.0, "waiting": 0.0}

    # Extract values from matches
    for match in matches:
        metric_type = match.group(1)
        if metric_type in ["running", "waiting"]:
            # print(match.group(2))
            metrics[metric_type] = math.floor(float(match.group(2)))

    return metrics["running"] + metrics["waiting"]


class VLLMServerManager:

    def __init__(
        self,
        model: str = "allenai/Llama-3.1-Tulu-3-8B-DPO",
        port: int = 8000,
        host: str = "0.0.0.0",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        registry_dir: str = "/gscratch/ark/graf/registry",
        quantization: str = None,
        max_new_tokens=800,
        max_prompt_length=1200,
        dtype="bfloat16",
        max_history=3600,
        hostname: str = "localhost",
    ):
        """
        Initialize VLLM server manager

        Args:
            model: Model name/path
            port: Server port
            host: Server host
            tensor_parallel_size: GPU parallel size
            gpu_memory_utilization: GPU memory usage (0-1)
            registry_dir: Directory for server registry
            max_model_len: Maximum sequence length
            quantization: Quantization method (e.g. 'awq')
        """
        self.model = model

        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_new_tokens + max_prompt_length
        self.quantization = quantization
        self.metrics_port = self.port
        self.dtype = dtype
        self.hostname = hostname

        # Initialize registry

        self.registry = ServerRegistry(
            store=FileSystemKVStore(registry_dir),
            max_history=max_history,
        )
        self.process = None
        self.should_run = True

    def start_server(self):
        """Start the vLLM server as a subprocess"""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",  #    entrypoints/api_server.py # openai.
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--dtype",
            self.dtype,
        ]

        # cmd.extend(["--prometheus-port", str(self.metrics_port)])  # or any other port

        print(cmd)

        if self.quantization:
            cmd.extend(["--quantization", self.quantization])

        print(f"vllm_server_{self.registry.server_id}.log")
        log_file = open(f"vllm_server_{self.registry.server_id}.log", "w")
        self.process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, universal_newlines=True
        )
        print(f"Started vLLM server with PID {self.process.pid}")

        # Register server with metadata
        metadata = {
            "model_path": self.model,
            "device_count": self.tensor_parallel_size,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "route": "v1/completions",
            "args": [
                "prompt",
                "model",
                "max_tokens",
                "temperature",
                "stop",
                "logprobs",
            ],
        }
        asyncio.run(
            self.registry.register_server(
                url="http://" + self.hostname, port=self.port, metadata=metadata
            )
        )

    def check_health(self):
        """Check if vLLM server is responding"""
        try:
            response = requests.get(f"http://localhost:{self.port}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:

            return False

    def heartbeat_loop(self):
        """Run heartbeat in a loop"""
        while self.should_run:
            if self.check_health():
                asyncio.run(self.registry.heartbeat(self.port))
                # print("Heartbeat sent. Status: healthy")
            else:
                print("Server unhealthy!")
            time.sleep(10)

    def cleanup(self):
        """Clean up resources"""
        self.should_run = False
        asyncio.run(self.registry.deregister())
        if self.process:
            self.process.terminate()
            self.process.wait()
        print("Server stopped and deregistered")

    def run(self):
        """Run server and monitoring"""
        try:
            self.start_server()
            print("Waiting for server to initialize...")
            time.sleep(30)  # Wait for model to load

            # Start heartbeat in background thread
            heartbeat_thread = threading.Thread(target=self.heartbeat_loop)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()

            # Wait for shutdown signal
            self.process.wait()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()


def main(
    model: str = "allenai/Llama-3.1-Tulu-3-8B-SFT",  # "allenai/Llama-3.1-Tulu-3-8B-DPO",  # allenai/Llama-3.1-Tulu-3-8B-SFT
    # port: int = 8000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    registry_dir: str = "/gscratch/ark/graf/registry",
    quantization: str = None,
    max_new_tokens=800,
    max_prompt_length=1200,
    dtype="bfloat16",
):
    """
    Run vLLM server with monitoring

    Args:
        model: Model name/path
        port: Server port
        host: Server host
        tensor_parallel_size: GPU parallel size
        gpu_memory_utilization: GPU memory usage (0-1)
        registry_dir: Directory for server registry
        max_model_len: Maximum sequence length
        quantization: Quantization method (e.g. 'awq')
    """
    manager = VLLMServerManager(
        model=model,
        port=random.randint(8000, 12000),
        host=host,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        registry_dir=registry_dir,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_prompt_length,
        dtype=dtype,
        hostname=f"{socket.gethostname()}.hyak.local",
    )
    manager.run()


if __name__ == "__main__":
    """python -m vllm.entrypoints.openai.api_server  --model allenai/Llama-3.1-Tulu-3-8B-DPO \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95
    """

    fire.Fire(main)
