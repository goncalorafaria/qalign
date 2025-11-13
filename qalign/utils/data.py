from datasets import load_dataset, Value

#from langchain.prompts import PromptTemplate

from functools import partial

from transformers import AutoTokenizer

import numpy as np
from datasets import Dataset

#DEFAULT_USER_PROMPT = PromptTemplate.from_template("Question: {content}\n")
#DEFAULT_AI_PROMPT = PromptTemplate.from_template("Answer: {content}\n")

## qalign
from qalign.utils.examples import MATH_EXAMPLARS, GSM8K_EXEMPLARS

from torch.utils.data import (
    Dataset,
    DataLoader,
)
from typing import List
from tqdm import tqdm

## quest

#DEFAULT_TEMPLATE = PromptTemplate.from_template("{prompt}")

class FlexiblePromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        result = self.template
        # Only replace variables that exist in the template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


DEFAULT_USER_PROMPT = FlexiblePromptTemplate("Question: {content}\n")
DEFAULT_AI_PROMPT = FlexiblePromptTemplate("Answer: {content}\n")
DEFAULT_TEMPLATE = FlexiblePromptTemplate("{prompt}")


def parsehh_to_template(
    text: str,
    br="\n\n",
    available_roles={"Human", "Assistant"},
    transfer_roles={
        "Human": "user",
        "Assistant": "assistant",
    },
):

    breaks = text.split(br)

    chlist = []
    for ch in breaks:
        try:
            subbreaks = ch.split(":")
            role = subbreaks[0]
            if role in available_roles:

                chlist.append(
                    {
                        "role": transfer_roles[role],  # .lower(),
                        "content": subbreaks[1].strip(),
                    }
                )
            else:

                if len(chlist) > 0:
                    chlist[-1]["content"] += br + ch

        except:
            pass

    return chlist


def general_process_data_chat(
    chat_template_prompt,
    tokenizer,
    **extra_data,
):

    return {
        "prompt": tokenizer.apply_chat_template(
            chat_template_prompt,  # [ {"role": "user", "content": ... }, {"role": "assistant", "content": ... }, ... ],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "chat_template_prompt": chat_template_prompt,
        **extra_data,
    }


def general_process_data_prompt(
    chat_template_prompt,
    tokenizer,
    user_prompt,
    ai_prompt,
    **extra_data,
):
    concatenated_prompt = tokenizer.bos_token

    for message in chat_template_prompt:
        role = message["role"]
        content = message["content"].strip()

        if role == "user":
            concatenated_prompt += user_prompt.format(content=content)
        elif role == "assistant":
            concatenated_prompt += ai_prompt.format(content=content)

        concatenated_prompt += "\n"  # Add extra newline between Q&A pairs

    return {
        "prompt": concatenated_prompt,
        "chat_template_prompt": chat_template_prompt,
        **extra_data,
    }


def general_process_data(chat_template_prompt, format="chat", **extra_data):

    if format == "chat":
        return general_process_data_chat(chat_template_prompt, **extra_data)
    else:
        return general_process_data_prompt(chat_template_prompt, **extra_data)


def processhh_data(entry, tokenizer, format="chat", prompt_template=None, **kwargs):

    br = "\n\n"
    available_roles = {"Human", "Assistant"}
    # breaks = entry["chosen"].split(br)

    chat_template = parsehh_to_template(
        entry["chosen"],
        br=br,
        available_roles=available_roles,
    )
    chat_template_reject = parsehh_to_template(
        entry["rejected"],
        br=br,
        available_roles=available_roles,
    )

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        tokenizer=tokenizer,
        format=format,
        answer=chat_template[-1]["content"],
        bad_answer=chat_template_reject[-1]["content"],
    )


def processgsm_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):
    if use_few_shot:
        gsm_messages = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(prompt=sample["question"]),
                },
                {"role": "assistant", "content": sample["cot_answer"]},
            ]
            for sample in GSM8K_EXEMPLARS
        ]
        # flatten
        gsm_messages = [item for sublist in gsm_messages for item in sublist]
    else:
        gsm_messages = []

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["question"]),
        },
        {
            "role": "assistant",
            "content": entry["answer"],
        },
    ]

    return general_process_data_chat(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=chat_template[-1]["content"],
    )


def processmath_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):

    if use_few_shot:
        math_messages = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(prompt=sample["question"]),
                },
                {"role": "assistant", "content": sample["cot_answer"]},
            ]
            for sample in MATH_EXAMPLARS
        ]
        # flatten
        math_messages = [item for sublist in math_messages for item in sublist]
    else:
        math_messages = []

    chat_template = math_messages + [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["problem"]),
        },
        {
            "role": "assistant",
            "content": entry["solution"],
        },
    ]

    # Get the final answer for evaluation
    # answer = get_last_math(chat_template[-1]["content"])
    # if answer is None:
    #    return None

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=entry["solution"],  # answer,
    )


def processdeepmath_data(
    entry, tokenizer, prompt_template, format="chat",  **kwargs
):
  
    math_messages = []

    chat_template = math_messages + [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["question"]),
        },
        {
            "role": "assistant",
            "content": entry["r1_solution_1"],
        },
    ]

    # Get the final answer for evaluation
    # answer = get_last_math(chat_template[-1]["content"])
    # if answer is None:
    #    return None

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=entry["r1_solution_1"],  # answer,
    )


def processalpaca_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):
    # import pdb; pdb.set_trace()

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["instruction"]),
        },
        {
            "role": "assistant",
            "content": entry["output"],
        },
    ]

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=chat_template[-1]["content"],
    )


def processifeval_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):
    # import pdb; pdb.set_trace()

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=entry["prompt"]),
        }
    ]

    return general_process_data(
        chat_template_prompt=chat_template,
        format=format,
        tokenizer=tokenizer,
        answer="no answer",
    )


def processmmlu_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):

    choice_options = ["A", "B", "C", "D"]

    l = entry["subject"].split("_")
    subject = ""
    for x in l:
        subject += " " + x

    instruction = f"Choose the correct answer to the following multiple-choice question about {subject}.\n\n"

    prompt = entry["question"]
    choices = entry["choices"]
    answer = int(entry["answer"])

    instruction += "Question: {}\n\n".format(prompt)

    for j in range(len(choices)):
        instruction += "{}). {}\n".format(choice_options[j], choices[j])

    instruction += "\nProvide your reasoning about the answer and finish your answer with the letter corresponding to the correct option (e.g., A, B, C, or D).\n\n"

    prompt_ans = "\nAnswer:"

    # if include_answer:
    prompt_ans += " {}\n\n".format(choice_options[answer])

    entry = {"prompt": instruction, "answer": prompt_ans}

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=instruction),
        },
        {
            "role": "assistant",
            "content": prompt_ans,
        },
    ]

    # Get the final answer for evaluation
    # answer = get_last_math(chat_template[-1]["content"])
    # if answer is None:
    #    return None

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=prompt_ans,  # answer,
    )


def processtruthfulqa_data(
    entry, tokenizer, prompt_template, format="chat", use_few_shot=False, **kwargs
):
    choices_key = "mc1_targets"

    choice_options = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    instruction = (
        "Choose the correct answer to the following multiple-choice question.\n\n"
    )

    prompt = entry["question"]
    choices = entry[choices_key]["choices"]
    answer = np.argmax(entry[choices_key]["labels"])

    # answer = int(entry["answer"])

    instruction += "Question: {}\n\n".format(prompt)

    for j in range(len(choices)):
        instruction += "{}). {}\n".format(choice_options[j], choices[j])

    instruction += "\nProvide your reasoning about the answer and finish your answer with the letter corresponding to the correct option (e.g., A, B, C, or D).\n\n"

    prompt_ans = "\nAnswer:"

    # if include_answer:
    prompt_ans += " {}\n\n".format(choice_options[answer])

    entry = {"prompt": instruction, "answer": prompt_ans}

    chat_template = [
        {
            "role": "user",
            "content": prompt_template.format(prompt=instruction),
        },
        {
            "role": "assistant",
            "content": prompt_ans,
        },
    ]

    return general_process_data(
        chat_template_prompt=chat_template[:-1],
        format=format,
        tokenizer=tokenizer,
        answer=prompt_ans,  # answer,
    )


def process_all_redux_data():

    subjects = [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "econometrics",
        "electrical_engineering",
        "formal_logic",
        "global_facts",
        "high_school_chemistry",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "human_aging",
        "logical_fallacies",
        "machine_learning",
        "miscellaneous",
        "philosophy",
        "professional_accounting",
        "public_relations",
        "virology",
        "conceptual_physics",
        "high_school_us_history",
        "astronomy",
        "high_school_geography",
        "high_school_macroeconomics",
        "professional_law",
    ]

    ds = []
    for subject in subjects:

        dsi = load_dataset("edinburgh-dawg/mmlu-redux", subject, split="test")
        ds.extend([{"subject": subject, **x} for x in dsi])

    dataset = Dataset.from_list(ds)

    return dataset


SUPPORTED_DATASETS = {
    "openai/gsm8k": ({"name": "socratic"}, processgsm_data, "openai/gsm8k"),
    "apple/GSM-Symbolic-p1": ({"name": "p1"}, processgsm_data, "apple/GSM-Symbolic"),
    "apple/GSM-Symbolic-p2": ({"name": "p2"}, processgsm_data, "apple/GSM-Symbolic"),
    "Anthropic/hh-rlhf": ({}, processhh_data, "Anthropic/hh-rlhf"),
    "lighteval/MATH": ({}, processmath_data, "lighteval/MATH"),
    "cais/mmlu": ({"name": "all"}, processmmlu_data, "cais/mmlu"),
    "truthfulqa/truthful_qa": (
        {"name": "multiple_choice"},
        processtruthfulqa_data,
        "truthfulqa/truthful_qa",
    ),
    "HuggingFaceH4/MATH-500": (
        {},
        processmath_data,
        "HuggingFaceH4/MATH-500",
    ),
    "tatsu-lab/alpaca_eval": (
        {},
        processalpaca_data,
        "tatsu-lab/alpaca_eval",
    ),
    "google/IFEval": (
        {},
        processifeval_data,
        "google/IFEval",
    ),
    "edinburgh-dawg/mmlu-redux": (
        {},
        processmmlu_data,
        "edinburgh-dawg/mmlu-redux",
        process_all_redux_data,
    ),
     "zwhe99/DeepMath-103K": (
        {},
        processdeepmath_data,
        "zwhe99/DeepMath-103K",
    ),
    # eval
    # "tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    # processifeval_data
}

# edinburgh-dawg/mmlu-redux


# ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")


def get_data_iterable(
    model_path: str,
    dataset_path: str,
    split: str = "test",
    n=None,
    prompt_template=DEFAULT_TEMPLATE,
    use_few_shot=False,
    format="chat",
    **kwargs,
):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if dataset_path in SUPPORTED_DATASETS:

        resources = SUPPORTED_DATASETS[dataset_path]

        if len(resources) == 3:
            kwargs, pfunc, dataset_path = SUPPORTED_DATASETS[dataset_path]

            ds = load_dataset(
                dataset_path,
                split=split,
                # desc=None,
                **kwargs,
            )
        else:
            kwargs, pfunc, dataset_path, load_func = SUPPORTED_DATASETS[dataset_path]
            ds = load_func()

        if "answer" in ds.column_names:
            ds = ds.cast_column("answer", Value("string"))

        # limit the size of the dataset

        if n is not None:
            ds = ds.select(list(range(n)))

        ds = ds.map(
            partial(
                pfunc,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                use_few_shot=use_few_shot,
                format=format,
            ),
            load_from_cache_file=False,
            # desc=None,  # Disable tqdm progress bar
        )

    else:
        ds = load_dataset(dataset_path, split=split, trust_remote_code=True, **kwargs)

        if n is not None:
            ds = list(ds)[:n]

    return list(ds)



class TextDataset(Dataset):
    def __init__(
        self,
        texts,
   
    ):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        return text




def get_loader(
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    use_tqdm: bool = False,
    max_length: int = 1024,
):
    ds = TextDataset(
        texts,
    )
    
    def collate_fn(batch):
    
        texts = [item for item in batch]
        
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True , 
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            #return_overflowing_tokens=True,
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    if use_tqdm:
        loader = tqdm(loader)

    return loader