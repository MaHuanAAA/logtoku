import os
import argparse
import re
import json
import pickle
from pprint import pprint


import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
from jinja2 import Template
import textwrap
from transformers import AutoTokenizer, LlamaForCausalLM



from datasets import load_dataset
import evaluate
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer, BleurtConfig

from utils import merge_jsonl_files
from metrics import topk, get_eu
from utils import seed_everything, color_print
from utils import save2jsonl, convert_float32_to_float, save2json_SE, load_jsonl_file
from utils import get_one_pass_metric, get_muti_pass_metric
from utils import process_decoded_str
from utils import calculate_and_save_metrics
from utils import calculate_and_save_metrics_llm
from transformers import BitsAndBytesConfig



metrics = [
    ("prob", None),
    ("entropy", None),
    ("au", 5),
    ("eu", 5),
    ("au_2", 2),
    ("eu_2", 2)
]
HF_NAMES = {
    'llama2_chat_7B': 'path/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'path/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'path/Llama-2-70b-chat-hf',
    'llama-3.2-1B-Instruct':'path/Llama-3.2-1B-Instruct',
    'llama-3.2-3B-Instruct':'path/Llama-3.2-3B-Instruct',
    'llama-3.1-8B-Instruct':'path/Llama-3.1-8B-Instruct',
    'llama-3.1-70B-Instruct':'path/llama_3_70b_hf'
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,          
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4",  
)
def get_prompt(MODEL, question):
    if 'llama2' in MODEL:
        # chat_template = textwrap.dedent("""\
        #     <s>[INST] 
        #     Answer the question concisely Q: {{ question }} 
        #     A:[/INST]""")
        chat_template = f"Answer the question concisely. Q: {question}" + " A:"
        template = Template(chat_template)
        generated_prompt = template.render(question=question)
    else:
        chat_template = textwrap.dedent("""\
            <|start_header_id|>system<|end_header_id|>

            Environment: ipython
            Tools: none

            <|eot_id|>
            <|start_header_id|>user<|end_header_id|

            Answer the question concisely. Q: {{ question }} A:<|eot_id|>
        """)
        template = Template(chat_template)
        generated_prompt = template.render(question=question) + "assistant\n\n"
    return generated_prompt

def main():
    parser = argparse.ArgumentParser()  # Create an argument parser
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')  
    parser.add_argument('--dataset_name', type=str, default='tqa')  

    parser.add_argument('--gene', type=int, default=0)
    parser.add_argument('--num_gene', type=int, default=1)

    parser.add_argument('--generate_gt', type=int, default=0)
    parser.add_argument('--thres_gt', type=float, default=0.5)
    parser.add_argument('--generate_gt_llm', type=int, default=0)

    parser.add_argument("--mode", type=str, default='one_pass', choices=['one_pass', 'muti_pass'])
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--gpuid", type=int, default=[0], nargs='+')

    args = parser.parse_args() 

    print(args.model_name, args.mode, args.temp, args.gpuid)
    print("hello world")
    if args.mode == 'one_pass':
        args.num_gene = 1
    elif args.mode == 'muti_pass':
        args.num_gene = 10

    MODEL = HF_NAMES[args.model_name]

    if args.dataset_name == "tqa":
            dataset = load_dataset("truthful_qa", 'generation')['validation']

    generation_config_one = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": False,
        "max_new_tokens": 64,
        "output_scores": True,
        "return_dict_in_generate": True
    }

    generation_config_muti = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "max_new_tokens": 64,
        "output_scores": True,
        "return_dict_in_generate": True,
        "temperature": args.temp,
        "top_p": 1.0
    }
    temper = generation_config_muti["temperature"]
    new_folder = "all_gene"
    save_one_pass = f"{new_folder}/{args.model_name}_one_pass_gene_{temper}.jsonl"
    save_muti_pass = f"{new_folder}/{args.model_name}_muti_pass_gene_{temper}.jsonl"
    save_eval = f"{new_folder}/{args.model_name}_eval_{temper}.jsonl"
    save_merge_one_pass = f"{new_folder}/{args.model_name}_one_pass_gene_merge_{temper}.jsonl"
    save_merge_muti_pass = f"{new_folder}/{args.model_name}_muti_pass_gene_merge_{temper}.jsonl"
    if args.gene:

        # set_device(args.model_name)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpuid))
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if '70B' in args.model_name:
            model = LlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, quantization_config=bnb_config,  
                                                            device_map="auto")
        else:
            model = LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                            device_map="auto")
        begin_index, end_index = 0, len(dataset)

        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]  # 取最后一个 token
        period_token_id = [newline_token_id, tokenizer.eos_token_id]
        

        generation_config_one["eos_token_id"]= period_token_id
        generation_config_muti["eos_token_id"]= period_token_id

        for i in range(begin_index, end_index):
            question_idx = i
            question = dataset[question_idx]['question']

            generated_prompt = get_prompt(args.model_name, question)

            prompt = tokenizer(generated_prompt, return_tensors='pt').input_ids.to(model.device)

            if args.mode == 'one_pass':
                generated = model.generate(prompt, **generation_config_one)
                decoded = tokenizer.decode(generated.sequences[0],skip_special_tokens=True)
                input_length = prompt.shape[-1]
                logits = generated.scores
                new_decoded = tokenizer.decode(generated.sequences[0][input_length:],skip_special_tokens=True)

                generated_tokens = generated.sequences[0][input_length:].clone()
                stop_phrases = ["Answer the question concisely", "Q:","\n\nQ:", "\nQ:","\"Q:\""," Q: "]
                clean_decoded, clean_generated_tokens_length = process_decoded_str(new_decoded, stop_phrases, tokenizer)
                metric_dict, logit_dict = get_one_pass_metric(logits, clean_generated_tokens_length, metrics, get_eu, topk)
                save2jsonl(new_decoded, metric_dict, logit_dict,save_one_pass, i)

            elif args.mode == 'muti_pass':
                metric_dict = {}
                for gen_iter in range(args.num_gene):
                    generated = model.generate(prompt, **generation_config_muti)
                    decoded = tokenizer.decode(generated.sequences[0],skip_special_tokens=True)
                    input_length = prompt.shape[-1]
                    logits = generated.scores
                    new_decoded = tokenizer.decode(generated.sequences[0][input_length:],skip_special_tokens=True)

                    generated_tokens = generated.sequences[0][input_length:].clone()
                    stop_phrases = ["Answer the question concisely", "Q:","\n\nQ:", "\nQ:","\"Q:\""]
                    # metric_dict = get_muti_pass_metric(logits, clean_generated_tokens, clean_generated_tokens_length, question, clean_decoded, gen_iter, metric_dict)
                    clean_decoded, clean_generated_tokens_length = process_decoded_str(new_decoded, stop_phrases, tokenizer)
                    metric_dict = get_muti_pass_metric(logits, generated_tokens, clean_generated_tokens_length, question, clean_decoded, gen_iter, metric_dict)
                    # print(metric_dict)
                save2json_SE(metric_dict, save_muti_pass, question_idx)


    elif args.generate_gt:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpuid))
        model = BleurtForSequenceClassification.from_pretrained('path/BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('path/BLEURT-20')
        length = len(dataset)
        one_pass_data = load_jsonl_file(save_one_pass)
        for i in range(length):
            if args.dataset_name == 'tqa':
                best_answer = dataset[i]['best_answer']
                correct_answer = dataset[i]['correct_answers']
                all_answers = [best_answer] + correct_answer
                predictions = one_pass_data[i].get("answer", [])  
                predictions = np.array([predictions], dtype=object) 
                calculate_and_save_metrics(i, predictions, all_answers, save_eval, model, tokenizer)
        merge_jsonl_files(save_one_pass, save_eval, save_merge_one_pass) # 得到了mostlikely的正确性和不确定性指标文件，可以用来计算auroc了

    elif args.generate_gt_llm:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpuid))
        Judge_model = 'path/Llama-3.1-8B-Instruct'
        model = LlamaForCausalLM.from_pretrained(Judge_model, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                        device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(Judge_model, trust_remote_code=True)
        length = len(dataset)
        one_pass_data = load_jsonl_file(save_one_pass)
        for i in range(length):
            if args.dataset_name == 'tqa':
                best_answer = dataset[i]['best_answer']
                correct_answer = dataset[i]['correct_answers']
                all_answers = [best_answer] + correct_answer
                question = dataset[i]['question']
            elif args.dataset_name == 'triviaqa':
                all_answers = dataset[i]['answer']['aliases']
                question = dataset[i]['question']
            predictions = one_pass_data[i].get("answer", [])
            predictions = np.array([predictions], dtype=object)
            calculate_and_save_metrics_llm(i, question, predictions, all_answers, save_eval, model, tokenizer)
        merge_jsonl_files(save_one_pass, save_eval, save_merge_one_pass)
if __name__ == '__main__':
    seed_everything(42)
    main()
                

