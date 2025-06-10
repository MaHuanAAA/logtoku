import os
import torch
import json
import numpy as np
from evaluate import load
YELLOW = "\033[93m"  # ANSI escape code for yellow text
RESET = "\033[0m"    # ANSI escape code to reset text formatting

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def color_print(prefix: str, text: str) -> None:

    highlighted_text = f"{YELLOW}{prefix} {RESET}{text}"

    print(highlighted_text)

def convert_float32_to_float(data):
    if isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_float32_to_float(v) for k, v in data.items()}
    else:
        return data
    
def load_jsonl_file(save_file_name):
    data = []
    with open(save_file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save2jsonl(new_decoded, metric_dict, logit_dict, save_file_name, i):
    os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
    with open(save_file_name, 'a') as output_file:
        prob_list = metric_dict['prob']
        entropy_list = metric_dict['entropy']
        au_list = metric_dict['au']
        eu_list = metric_dict['eu']
        au_2_list = metric_dict['au_2']
        eu_2_list = metric_dict['eu_2']

        prob_list = convert_float32_to_float(prob_list)
        entropy_list = convert_float32_to_float(entropy_list)
        au_list = convert_float32_to_float(au_list)
        eu_list = convert_float32_to_float(eu_list)
        au_2_list = convert_float32_to_float(au_2_list)
        eu_2_list = convert_float32_to_float(eu_2_list)

        final_output = {
            'question_id': i,
            'answer': new_decoded,
            'prob': prob_list,
            'entropy': entropy_list,
            'au': au_list,
            'eu': eu_list,
            'au_2': au_2_list,
            'eu_2': eu_2_list,
            'Token_logit_dict': logit_dict
        }
        output_file.write(json.dumps(final_output, ensure_ascii=False) + '\n')

def save2json_SE(gen_data, save_file_name, question_id):
    """
    Save/update multi-round iteration results to a JSON file.
    :param gen_data: Dictionary of data for the current iteration {gen_iter: (new_decoded, prob)}.
    :param save_file_name: Path to the JSON file.
    :param question_id: ID of the current question.
    """
    # Read existing data if the file exists
    if os.path.exists(save_file_name):
        with open(save_file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Build the data structure for the current question
    question_entry = {
        'question_id': question_id,
        'generations': gen_data  # Directly use the passed {gen_iter: (text, prob)} dictionary
    }
    # Update or add data for the current question
    existing_data[str(question_id)] = question_entry  # Use string format to ensure consistent JSON key type

    # Write the updated data back to the file
    with open(save_file_name, 'w') as f:
        json.dump(existing_data, f, indent=2)

def calculate_and_save_metrics(idx, predictions, all_answers, save_file_name, model, tokenizer):
    result = {"question_id": idx}

    rouge = load('rouge')
    rouge_results = np.zeros((len(all_answers), len(predictions)))
    for anw in range(len(all_answers)):
        results = rouge.compute(predictions=predictions, references=[all_answers[anw]] * len(predictions), use_aggregator=False)
        rouge_results[anw] = results['rougeL']
    result["rouge"] = np.max(rouge_results, axis=0).tolist()

    all_answers = [answer if answer.endswith('.') else answer + '.' for answer in all_answers]

    model.eval()

    predictions = predictions.tolist()[0]
    with torch.no_grad():
        inputs = tokenizer(all_answers, [predictions] * len(all_answers), padding='longest', return_tensors='pt')
        
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        scores = model(**inputs).logits.flatten()

        scores = scores.cpu().tolist()

    result["bleurt"] = max(scores)

    with open(save_file_name, 'a') as f:
        f.write(json.dumps(result) + '\n')

def calculate_and_save_metrics_llm(idx, question, predictions, all_answers, save_file_name, model, tokenizer):
    result = {"question_id": idx}

    rouge = load('rouge')
    rouge_results = np.zeros((len(all_answers), len(predictions)))
    for anw in range(len(all_answers)):
        results = rouge.compute(predictions=predictions, references=[all_answers[anw]] * len(predictions), use_aggregator=False)
        rouge_results[anw] = results['rougeL']
    result["rouge"] = np.max(rouge_results, axis=0).tolist()

    model.eval()
    predictions = predictions.tolist()[0]
    
    messages = [
        {"role": "system", "content": "Your task is to determine if the provided answer is true or false based solely on the ground truth answers given to you in the format [’answer 1’, ’answer 2’, ...]. DO NOT rely on your memory; only use the information provided after this instruction. Respond with 1 if the predicted answer is correct, which means semantically consistent with any of the ground truth answers, otherwise respond with 0. Respond with just 0 or 1, and DO NOT include anything else in your response. This is the only instruction you need to follow."},
        {
            "role": "user",
            "content": "Input: Who is elected as the vice president of india in 2017?\nGround Truth: [‘Venkaiah Naidu’, ‘Muppavarapu Venkaiah Naidu’]\nProvided Answer: M. Venkaiah Naidu"
        },
        {"role": "assistant", "content": "1"},
        {
            "role": "user", 
            "content": "Input: who sings you are a magnet and i am steel?\nGround Truth: [‘Walter Egan’]\nProvided Answer: The song ‘You Are a Magnet and I Am Steel’ is performed by the band The 1975."
        },
        {"role": "assistant", "content": "0"}
    ]

    current_prompt = {
        "role": "user",
        "content": f"Input: {question}\nGround Truth: {all_answers}\nProvided Answer: {predictions}"
    }
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages + [current_prompt],
        tokenize=False,
        add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        input_len = inputs.input_ids.shape[-1]
        judge_result = 0
        clean_decoded_output = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
        last_response = decoded_output.split("assistant\n")[-1].strip()
        # print("*****************************")
        # print(f"LLM judgment result: {clean_decoded_output}")
        # print("*****************************")
        if "1" in last_response:
            judge_result = 1
        elif "0" in last_response:
            judge_result = 0
        else:
            print(f"Abnormal output: {decoded_output}")
            judge_result = 0

    result["llm_judge"] = judge_result

    with open(save_file_name, 'a') as f:
        f.write(json.dumps(result) + '\n')

def process_decoded_str(new_decoded, stop_phrases, tokenizer):
    """Process the decoded string, truncate by stop phrases, and return the clean string and token length"""
    min_pos = None
    
    # Check the position of all stop phrases
    for phrase in stop_phrases:
        pos = new_decoded.find(phrase)
        if pos != -1:  # Match found
            if (min_pos is None) or (pos < min_pos):
                min_pos = pos

    # Perform truncation
    min_pos = min_pos if min_pos is not None else len(new_decoded)
    clean_decoded = new_decoded[:min_pos] if min_pos is not None else new_decoded

    # Encode and calculate the length (without adding special tokens)
    clean_tokens = tokenizer.encode(clean_decoded, add_special_tokens=False)
    return clean_decoded, len(clean_tokens)

def get_one_pass_metric(logits, clean_generated_tokens_length, metrics, get_eu, topk):
    """
    Process logits, calculate various metrics, and extract topk information from the logits.
    
    Args:
        
        clean_generated_tokens_length (int): Length of the generated tokens.
        metrics (list): List of metrics and their parameters.
        get_eu (function): Function to retrieve metric values.
        topk (function): Function to extract topk values.
        
    Returns:
        tuple: A tuple containing metric_dict and logit_dict.
    """
    metric_dict = {}

    sequence_length = len(logits)
    for idx_l in range(min(clean_generated_tokens_length, sequence_length)):
        logit = logits[idx_l]
        logit = logit.cpu().numpy()
        
        for metric, k in metrics:
            if metric not in metric_dict:
                metric_dict[metric] = []
            eu = get_eu(metric, k)
            metric_dict[metric].append(eu(logit[0]))

    logit_dict = {}
    logit_start_idx = 0
    logit_end_idx = min(clean_generated_tokens_length, sequence_length)
    ii = 0

    # Extract topk information from logits
    for idx_ll in range(logit_start_idx, logit_end_idx):
        logit = logits[idx_ll]
        logit = logit.cpu().numpy()
        
        top_k = 10
        top_values, top_indices = topk(logit[0], top_k)
        
        logit_dict[ii] = {'top_values': top_values, 'top_indices': top_indices}
        ii += 1

    # Convert top_values and top_indices in logit_dict to list
    for key in logit_dict:
        logit_dict[key]['top_values'] = logit_dict[key]['top_values'].tolist()
        logit_dict[key]['top_indices'] = logit_dict[key]['top_indices'].tolist()

    return metric_dict, logit_dict

def get_muti_pass_metric(logits, clean_generated_tokens, clean_generated_tokens_length, question, clean_decoded, gen_iter, metric_dict):
    """
    Calculate the negative log likelihood (NLL) and update the results.
    
    Args:
        logits (torch.Tensor): Model's output logits, shape (sequence_length, vocab_size).
        clean_generated_tokens (torch.Tensor): Generated token ID sequence.
        clean_generated_tokens_length (int): Length of the generated token sequence.
        question (str): The original input question text.
        clean_decoded (str): The decoded generated text.
        gen_iter (int): The current generation iteration number.
        metric_dict (dict): Dictionary to store the results of each generation iteration.
        
    Returns:
        dict: Updated `metric_dict`, containing the NLL value and other information for the current iteration.
    """
    log_likelihood = 0.0
    sequence_length = len(logits)
    logit_end_idx = min(clean_generated_tokens_length, sequence_length)
    for step in range(logit_end_idx):
        cur_logits = logits[step]
        log_probs = torch.log_softmax(cur_logits, dim=-1)
        next_token_id = clean_generated_tokens[step]
        log_likelihood += log_probs[0, next_token_id].item()

    nll = -log_likelihood
    ln_nll = nll / clean_generated_tokens_length if clean_generated_tokens_length > 0 else 0.0

    # Update iteration results
    metric_dict[gen_iter] = {
        "text": question + " " + clean_decoded,
        "nll": nll,
        "ln_nll": ln_nll,
        "ans": clean_decoded,
        "ques": question
    }

    return metric_dict

def merge_jsonl_files(file1, file2, output_file):
    """
    Merge two JSONL files by combining data based on question_id.

    Args:
        file1 (str): The path to the first JSONL file.
        file2 (str): The path to the second JSONL file.
        output_file (str): The path to the output file where the merged data will be saved.
    """
    # Read data from the first file
    data1 = {}
    with open(file1, 'r') as f:
        for line in f:
            item = json.loads(line)
            data1[item['question_id']] = item

    # Read data from the second file
    data2 = {}
    with open(file2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data2[item['question_id']] = item

    # Merge data from both files
    merged_data = {}
    for question_id in set(data1.keys()).union(data2.keys()):
        merged_item = {"question_id": question_id}
        if question_id in data1:
            merged_item.update(data1[question_id])
        if question_id in data2:
            merged_item.update(data2[question_id])
        merged_data[question_id] = merged_item

    # Write the merged data to the output file
    with open(output_file, 'w') as f:
        for question_id in merged_data:
            f.write(json.dumps(merged_data[question_id]) + '\n')

    print(f"Merge complete. The result has been saved to {output_file}")

def merge_jsonl_files_se(file1, file2, output_file):
    """
    Merge two JSONL files by combining data based on question_id (integer).

    Args:
        file1 (str): The path to the first JSONL file (question_id is a string number).
        file2 (str): The path to the second JSONL file.
        output_file (str): The path to the output file where the merged data will be saved.
    """
    # Read data from the first file and convert question_id to an integer
    data1 = {}
    with open(file1, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Convert question_id to integer
            item["question_id"] = int(item["question_id"])
            data1[item['question_id']] = item

    # Read data from the second file (ensure question_id is an integer)
    data2 = {}
    with open(file2, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Ensure the question_id in the second file is also an integer
            item["question_id"] = int(item["question_id"]) if isinstance(item["question_id"], str) else item["question_id"]
            data2[item['question_id']] = item

    # Merge data based on integer question_id
    merged_data = {}
    for question_id in set(data1.keys()).union(data2.keys()):
        merged_item = {"question_id": question_id}
        if question_id in data1:
            merged_item.update(data1[question_id])
        if question_id in data2:
            merged_item.update(data2[question_id])
        merged_data[question_id] = merged_item

    # Write the merged data to the output file
    with open(output_file, 'w') as f:
        for question_id in sorted(merged_data.keys()):
            f.write(json.dumps(merged_data[question_id]) + '\n')

    print(f"Merge complete. The result has been saved to {output_file}")
