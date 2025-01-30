import os
import torch
import argparse
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import json
import re
from itertools import islice
import torch.nn.functional as F
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import numpy as np
import random
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from itertools import islice

parser = argparse.ArgumentParser(description="Run the emotion classification model.")
parser.add_argument("--exp", type=str, default="llama2_7b", help="Experiment type (e.g., llama2_7b, llama2_13b)")
parser.add_argument("--gpuid", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
parser.add_argument("--quantize", type=bool, default=True, help="Enable or disable quantization (default: True)")

args = parser.parse_args()

gpuid_list = [int(id) for id in args.gpuid.split(",")]

warnings.simplefilter("ignore", UserWarning)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(2)

model_dict = {
    "llama2_7b": "/path/to/your/models/Llama-2-7b-chat-hf",
    "llama2_13b": "/path/to/your/models/Llama-2-13b-chat-hf",
    "llama2_70b": "/path/to/your/models/Llama-2-70b-chat-hf",
    "llama3_1_8b": "/path/to/your/models/Llama-3.1-8B-Instruct",
    "llama3_2_3b": "/path/to/your/models/Llama-3.2-3B-Instruct",
    "llama3_70b": "/path/to/your/models/Llama-3.1-70B-Instruct"
}

config = {
    "exp": args.exp,
    "gpuid": gpuid_list,
    "quantize": args.quantize
}

with open("end/configs/prompt.json") as f:
    prompt_config = json.load(f)

def create_prompt(sentence, config):
    if "llama2" in config["exp"]:
        prompt = prompt_config["prompt_llama2"].format(sentence)
    else:
        prompt = prompt_config["prompt_default"].format(sentence)
    return prompt

def load_model(config, model_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config["gpuid"]))
    
    base_model = model_dict[config["exp"]]
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    ) if config["quantize"] else None
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    model.eval()
    
    return tokenizer, model

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
save_file_name = f"outputs/{config['exp']}.jsonl"
os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

tokenizer, model = load_model(config, model_dict)

emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
emotion_to_idx = {e: tokenizer.encode(e, add_special_tokens=False)[0] for e in emotions}

generation_config = GenerationConfig(
    max_new_tokens=1,
    do_sample=False,
    repetition_penalty=1.0,
    output_scores=True,
    return_dict_in_generate=True
)

with open(save_file_name, "w", encoding="utf-8") as f_out:
    for split in ['test']:
        print(f"Processing split: {split}")
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            inputs = tokenizer(
                create_prompt(example['Tweet']),
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            emotion_logits = {
                e: outputs.scores[0][0, tid].item()
                for e, tid in emotion_to_idx.items()
            }
            
            f_out.write(json.dumps({
                "ID": example['ID'],
                "Tweet": example['Tweet'],
                "Emotion_Vector": {e: example[e] for e in emotions},
                "Emotion_Logits": emotion_logits
            }, ensure_ascii=False) + "\n")
