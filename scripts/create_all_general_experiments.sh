#!/bin/bash


MODELPATH=allenai/Llama-3.1-Tulu-3-8B-SFT
DIR="outputs/"

TEMP=1.0
N=128

#### QUEST
RM_PATH=allenai/Llama-3.1-Tulu-3-8B-RM
BETA=0.5
STEPS=1024
N=500
DATASET="HuggingFaceH4/MATH-500"
PROMPT="Solve the following math problem step-by-step: {prompt}\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type contextual --split "test" --num_chains 1 --batch_size 32 --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"

N=1024
DATASET="openai/gsm8k"
PROMPT="Solve the following grade school math problem step-by-step: {prompt}"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type contextual --split "test" --num_chains 1 --batch_size 32 --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"

N=541
DATASET="google/IFEval"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type contextual --split "train" --num_chains 1 --batch_size 32 --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 

N=3000
DATASET="edinburgh-dawg/mmlu-redux"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type contextual --split "test" --num_chains 1 --batch_size 32 --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 

N=817
DATASET="truthfulqa/truthful_qa"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "quest-rlhf" --reward_type contextual --split "test" --num_chains 1 --batch_size 32 --beta $BETA --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH  --reward_model_path $RM_PATH  --reward_model_batch_size 32 --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 


## ANCESTRAL 
DATASET="HuggingFaceH4/MATH-500"
PROMPT="Solve the following math problem step-by-step: {prompt}\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test" --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"

DATASET="openai/gsm8k"
PROMPT="Solve the following grade school math problem step-by-step: {prompt}"

TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test" --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"

DATASET="google/IFEval"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "train"  --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET

DATASET="edinburgh-dawg/mmlu-redux"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test"  --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET

## DPO ANCESTRAL 
MODELPATH=allenai/Llama-3.1-Tulu-3-8B-DPO

DATASET="HuggingFaceH4/MATH-500"
PROMPT="Solve the following math problem step-by-step: {prompt}\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test" --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"

DATASET="openai/gsm8k"
PROMPT="Solve the following grade school math problem step-by-step: {prompt}"

TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test"  --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET --prompt_template "$PROMPT"


DATASET="google/IFEval"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "train"  --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 

DATASET="edinburgh-dawg/mmlu-redux"
TOKENIZERS_PARALLELISM=false python launch_experiment.py --variant "ancestral" --split "test"  --batch_size 1 --steps $STEPS --temperature $TEMP --n $N --save_path $DIR  --gpu_memory_utilization 0.95 --model_path $MODELPATH --max_new_tokens 800 --max_prompt_length 1200 --dataset_path $DATASET 


# Loop through all files in the directory
for file in "$DIR/*; do
    # Skip if it's a directory
    if [ -f "$file" ]; then
        # Get just the filename without path
        fileid=$(basename "$file")
        
        echo "Processing: $fileid"
        
        # Launch Python script with the filename as experiment_name
        python resume_experiment.py --experiment_name "$fileid" --save_path "$DIR" 
        
        echo ""
    fi
done

echo "All experiments have been processed."