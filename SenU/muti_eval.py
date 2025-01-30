import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from rouge import Rouge
import itertools
import json
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import pickle
import argparse
from sklearn.metrics import roc_curve, auc

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        # Load DeBERTa tokenizer and model for sequence classification
        self.tokenizer = AutoTokenizer.from_pretrained("/path//models/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/path//models/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        # Tokenize the input texts and check implication between text1 and text2
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 implies text2 (text1 -> text2).
        # For example: check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # DeBERTa returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()

        return prediction

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):
        # Check if text1 and text2 are semantically equivalent
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            # For strict entailment, both implications must be entailment (2)
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            # For non-strict entailment, we allow neutral (1) and entailment (2)
            implications = [implication_1, implication_2]
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialize all ids with -1
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned
        if semantic_set_ids[i] == -1:
            # If not, assign it the next_id
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Check for equivalence with other strings
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids

def predictive_entropy(log_probs):
    """Compute MC estimate of entropy."""
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy

def predictive_entropy_rao(log_probs):
    """Compute Rao's entropy using sequence likelihoods."""
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from cluster assignments."""
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id using Log-Sum-Exp."""
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` for the active `uid`
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # Normalize the log likelihoods and compute log-sum-exp
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id

def calculate_lexical_similarity(texts):
    # Ensure there are at least 2 texts to compute similarity
    if len(texts) < 2:
        raise ValueError("At least 2 texts are required to calculate similarity")
    
    # Initialize Rouge calculator (automatically handles case and whitespace)
    rouge = Rouge()
    
    total_score = 0.0
    valid_pairs = 0
    
    # Generate all unique text pair combinations (i < j)
    for i, j in itertools.combinations(range(len(texts)), 2):
        try:
            scores = rouge.get_scores(texts[i], texts[j])
            total_score += scores[0]['rouge-l']['f']
            valid_pairs += 1
        except Exception as e:
            print(f"Error calculating similarity for pair {i}-{j}: {str(e)}")
    
    if valid_pairs == 0:
        return 0.0
    
    return total_score / valid_pairs

# Argument parsing for model configuration
parser = argparse.ArgumentParser() 
parser.add_argument("--model", type=str, default="7b", help="Model name")
parser.add_argument("--temp", type=float, default=0.5, help="Temperature")
parser.add_argument("--gpuid", type=int, default=0, help="GPU ID")
parser.add_argument("--strict", type=bool, default=False, help="Strict entailment")
parser.add_argument("--one_eval_path", type=str, default="all_gene", help="One evaluation dictionary path")
parser.add_argument("--muti_eval_path", type=str, default="muti_eval", help="Multiple evaluation dictionary path")
args = parser.parse_args()

model = args.model
temp = args.temp
one_eval_dict = args.one_eval_dict
muti_eval_path = args.muti_eval_path
strict_entailment = args.strict
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entailment_model = 'deberta'

if entailment_model == 'deberta':
    entailment_model = EntailmentDeberta()

# Continue the rest of the code...
# Paths for the evaluation files for different model sizes
all_gene_dict = {
    "7b": f"{one_eval_dict}/llama2_chat_7B_muti_pass_gene_{temp}.jsonl",
    "13b": f"{one_eval_dict}/llama2_chat_13B_muti_pass_gene_{temp}.jsonl",
    "70b": f"{one_eval_dict}/llama2_chat_70B_muti_pass_gene_{temp}.jsonl",
    "3b": f"{one_eval_dict}/llama-3.2-3B-Instruct_muti_pass_gene_{temp}.jsonl",
    "8b": f"{one_eval_dict}/llama-3.1-8B-Instruct_muti_pass_gene_{temp}.jsonl",
    "70b_3": f"{one_eval_dict}/llama3_chat_70B_muti_pass_gene_{temp}.jsonl"
}

# Paths for evaluation files for different model sizes
one_eval_dict = {
    "7b": f"{one_eval_dict}/llama2_chat_7B_eval_{temp}.jsonl",
    "13b": f"{one_eval_dict}/llama2_chat_13B_eval_{temp}.jsonl",
    "70b": f"{one_eval_dict}/llama2_chat_70B_eval_{temp}.jsonl",
    "3b": f"{one_eval_dict}/llama-3.2-3B-Instruct_eval_{temp}.jsonl",
    "8b": f"{one_eval_dict}/llama-3.1-8B-Instruct_eval_{temp}.jsonl",
    "70b_3": f"{one_eval_dict}/llama3_chat_70B_eval_{temp}.jsonl"
}

# Paths for multi-pass evaluation files
muti_eval_dict = {
    "7b": f"./{muti_eval_path}/llama2_chat_7B_muti_pass_gene_{temp}_metrics.jsonl",
    "13b": f"./{muti_eval_path}/llama2_chat_13B_muti_pass_gene_{temp}_metrics.jsonl",
    "3b": f"./{muti_eval_path}/llama-3.2-3B-Instruct_muti_pass_gene_{temp}_metrics.jsonl",
    "8b": f"./{muti_eval_path}/llama-3.1-8B-Instruct_muti_pass_gene_{temp}_metrics.jsonl",
    "70b": f"./{muti_eval_path}/llama2_chat_70B_muti_pass_gene_{temp}_metrics.jsonl",
    "70b_3": f"./{muti_eval_path}/llama3_chat_70B_muti_pass_gene_{temp}_metrics.jsonl"
}

# Paths for the final evaluation files
muti_eval_final_dict = {
    "7b": f"./{muti_eval_path}/llama2_chat_7B_muti_pass_gene_{temp}_final.jsonl",
    "13b": f"./{muti_eval_path}/llama2_chat_13B_muti_pass_gene_{temp}_final.jsonl",
    "3b": f"./{muti_eval_path}/llama-3.2-3B-Instruct_muti_pass_gene_{temp}_final.jsonl",
    "8b": f"./{muti_eval_path}/llama-3.1-8B-Instruct_muti_pass_gene_{temp}_final.jsonl",
    "70b": f"./{muti_eval_path}/llama2_chat_70B_muti_pass_gene_{temp}_final.jsonl",
    "70b_3": f"./{muti_eval_path}/llama3_chat_70B_muti_pass_gene_{temp}_final.jsonl"
}

muti_eval_dict_save = muti_eval_dict[model]
muti_eval_final_dict_path = muti_eval_final_dict[model]

# Check if the intermediate file already exists
if not os.path.exists(muti_eval_dict_save):
    print(f"Starting to process and generate intermediate file: {muti_eval_dict_save}")
    file_path = all_gene_dict[model]
    
    # Read the data from the model-specific file
    with open(file_path, 'r') as file:
        data = json.load(file)

    result_dict = {}
    for question_id, content in data.items():
        # Extract the generations and related metrics
        generations = content["generations"]
        texts = [gen["text"] for gen in generations.values()]
        nlls = [gen["nll"] for gen in generations.values()]
        ln_nlls = [gen["ln_nll"] for gen in generations.values()]
        anss = [gen["ans"] for gen in generations.values()]
        
        # Store the extracted data
        result_dict[question_id] = {
            "texts": texts,
            "nlls": nlls,
            "ln_nlls": ln_nlls,
            "ans": anss
        }

    metric_dict = {}

    # Process each question in the result dictionary
    for question_id, values in tqdm(result_dict.items(), desc="Processing questions"):
        texts = values["texts"]
        nlls = values["nlls"]
        ln_nlls = values["ln_nlls"]
        anss = values["ans"]
        
        # Calculate log likelihoods and length-normalized log likelihoods (per sentence)
        lls = [-nll for nll in nlls]
        ln_lls = [-ln_nll for ln_nll in ln_nlls]

        metric_dict[question_id] = {"question_id": question_id}

        # Calculate semantic set IDs for texts
        semantic_set_ids = get_semantic_ids(texts, entailment_model, strict_entailment)
        metric_dict[question_id]["semantic_set_ids"] = semantic_set_ids
        
        # Calculate discrete semantic entropy
        discrete_SE = cluster_assignment_entropy(semantic_set_ids)
        metric_dict[question_id]["discrete_SE"] = discrete_SE
        
        # Normalize log likelihoods using the semantic set IDs
        log_likelihood_per_semantic_id_ln = logsumexp_by_id(semantic_set_ids, ln_lls, agg='sum_normalized')
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_set_ids, lls, agg='sum_normalized')

        # Compute Rao's entropy using sequence likelihoods
        rao_SE = predictive_entropy_rao(log_likelihood_per_semantic_id)
        rao_SE_ln = predictive_entropy_rao(log_likelihood_per_semantic_id_ln)
        metric_dict[question_id]["rao_SE"] = rao_SE
        metric_dict[question_id]["rao_SE_ln"] = rao_SE_ln

        # Calculate Rao's entropy for sequence likelihoods
        rao_PE_ln = predictive_entropy_rao(log_likelihood_per_sentence_id_ln)
        rao_PE = predictive_entropy_rao(log_likelihood_per_semantic_id)
        metric_dict[question_id]["rao_PE_ln"]= rao_PE_ln
        metric_dict[question_id]["rao_PE"]= rao_PE

        # Compute lexical similarity of answers (optional)
        lex_Sim = calculate_lexical_similarity(anss)
        metric_dict[question_id]["lex_Sim_neg"] = -lex_Sim

    # Save the intermediate results to the file
    with open(muti_eval_dict_save, 'w', encoding='utf-8') as f:
        for item in metric_dict.values():
            item = {k: (v if v != -0.0 else 0.0) for k, v in item.items()}
            f.write(json.dumps(item) + '\n')
    print(f"Successfully generated intermediate file: {muti_eval_dict_save}")
else:
    print(f"Intermediate file already exists, skipping processing step: {muti_eval_dict_save}")

# Step 2: Check and generate the final file
if not os.path.exists(muti_eval_final_dict_path):
    print(f"Starting to merge and generate final file: {muti_eval_final_dict_path}")
    from utils import merge_jsonl_files_se
    correct_file = one_eval_dict[model]
    # Merge the evaluation files into the final file
    merge_jsonl_files_se(muti_eval_dict_save, correct_file, muti_eval_final_dict_path)
    print(f"Successfully generated final file: {muti_eval_final_dict_path}")
else:
    print(f"Final file already exists, skipping merge step: {muti_eval_final_dict_path}")

# Execute analysis on the final evaluation file
def load_metrics_data(file_path):
    """Load and preprocess metric data"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Remove 'rouge' field from data
    for item in data:
        item.pop('rouge', None)
    return data

def calculate_simple_auroc(data, metric_name):
    """Calculate AUROC for the specified metric"""
    # Generate binary labels based on 'bleurt' threshold
    labels = np.array([x['bleurt'] > 0.5 for x in data])
    
    # Extract metric scores
    scores = np.array([-x.get(metric_name, np.nan) for x in data])
    
    # Filter out invalid values
    valid_mask = ~np.isnan(scores)
    if valid_mask.sum() == 0:
        return np.nan
    
    # Compute AUROC
    fpr, tpr, _ = roc_curve(labels[valid_mask], scores[valid_mask])
    return auc(fpr, tpr)

def analyze_simple_metrics(file_path):
    """Main analysis function"""
    data = load_metrics_data(file_path)
    
    # Identify numeric metrics (excluding specific fields)
    exclude_fields = {'question_id', 'bleurt', 'rouge'}
    metric_names = [k for k in data[0].keys() 
                   if k not in exclude_fields 
                   and isinstance(data[0][k], (int, float))]
    
    print("\n{:<20} {:<10}".format('Metric Name', 'AUROC'))
    print("-" * 30)
    
    results = {}
    for metric in metric_names:
        auroc = calculate_simple_auroc(data, metric)
        results[metric] = auroc
        print("{:<20} {:.4f}".format(metric, auroc))
    
    # Calculate baseline accuracy
    baseline_acc = np.mean([x['bleurt'] > 0.5 for x in data])
    print("-" * 30)
    print("{:<20} {:.4f}".format('Baseline ACC', baseline_acc))
    
    return results

analyze_simple_metrics(muti_eval_final_dict_path)
