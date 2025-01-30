import json
import numpy as np
from sklearn.metrics import roc_curve, auc

def load_processed_data(file_path):
    """Load and process data from a JSONL file"""
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_data.append(data)
    return processed_data

def rank_fun(entry, indicator, k=25, p=5):
    """Function to calculate the aggregated score"""
    def top_k_mean(values, k):
        """Calculate the mean of the top k elements"""
        if values is None or len(values) == 0:
            return np.nan
        values = np.array(values)
        if len(values) <= k:
            return np.mean(values)
        top_k = np.partition(values, -k)[-k:]
        return np.mean(top_k)
        
    def all_mean(values):
        """Calculate the mean of all elements"""
        if values is None or len(values) == 0:
            return np.nan
        return np.mean(values)

    try:
        if indicator == "topk_prob":
            probs = entry.get("prob", [])
            if not probs:
                return np.nan
            return -top_k_mean(-np.log(probs), k)
        elif indicator == "topk_entropy":
            entropy = entry.get("entropy", [])
            return -top_k_mean(entropy, k) if entropy else np.nan
        elif indicator == "topk_logtu":
            eu = entry.get("eu_2", [])
            au = entry.get("au_2", [])
            if eu and au:
                combined = np.array(eu) * np.array(au)
                return -top_k_mean(combined, k)
            return np.nan
        elif indicator == "avg_entropy":
            entropy = entry.get("entropy", [])
            return -all_mean(entropy) if entropy else np.nan
        elif indicator == "avg_prob":
            probs = entry.get("prob", [])
            if not probs:
                return np.nan
            return -all_mean(-np.log(probs))
        elif indicator == "avg_logtu":
            eu = entry.get("eu_2", [])
            au = entry.get("au_2", [])
            if eu and au:
                combined = np.array(eu) * np.array(au)
                return -all_mean(combined)
            return np.nan
        else:
            print(f"Unrecognized indicator: {indicator}")
            return np.nan
    except Exception as e:
        print(f"Error calculating indicator {indicator}: {str(e)}")
        return np.nan

def calculate_auroc(processed_data, indicator):
    """Calculate AUROC (Area Under the ROC Curve)"""
    labels = np.array([entry['bleurt'] > 0.50 for entry in processed_data])
    scores = np.array([rank_fun(entry, indicator) for entry in processed_data])
    
    # Filter out invalid values
    valid_mask = ~np.isnan(scores)
    valid_labels = labels[valid_mask]
    valid_scores = scores[valid_mask]
    
    if len(valid_labels) == 0:
        print(f"Warning: No valid data for indicator {indicator}")
        return 0.0
    
    # Compute AUROC
    fpr, tpr, _ = roc_curve(valid_labels, valid_scores)
    return auc(fpr, tpr)

def analyze_multiple_models(model_paths):
    """Analyze the performance of multiple models and print a table"""
    metrics = ['avg_prob','avg_entropy','avg_logtu','topk_prob', 'topk_entropy', 'topk_logtu','topp_prob', 'topp_entropy', 'topp_logtu', 'Baseline ACC']
    results = {model_name: {} for model_name in model_paths.keys()}
    
    # Iterate over each model and calculate metrics
    for model_name, file_path in model_paths.items():
        processed_data = load_processed_data(file_path)
        
        # Calculate AUROC for each metric
        for metric in metrics[:-1]:  # Exclude Baseline ACC
            auroc = calculate_auroc(processed_data, metric)
            results[model_name][metric] = auroc
        
        # Calculate Baseline ACC
        baseline_acc = np.mean([entry['bleurt'] > 0.50 for entry in processed_data])
        results[model_name]['Baseline ACC'] = baseline_acc
    
    # Print the table
    print("{:<13}".format("Metric"), end=" | ")
    for model_name in model_paths.keys():
        print("{:<10}".format(model_name), end=" | ")
    print("\n" + "-" * (15 + 12 * len(model_paths)))
    
    for metric in metrics:
        print("{:<15}".format(metric), end=" | ")
        for model_name in model_paths.keys():
            print("{:<10.4f}".format(results[model_name][metric]), end=" | ")
        print()
    
    print("=" * (15 + 12 * len(model_paths)))


model_paths_dict = {
    "llama2-7b" : "all_gene/llama2-7b_one_pass_gene_merge_0.5.jsonl",
    "llama2-13b": "all_gene/llama2-13b_one_pass_gene_merge_0.5.jsonl",
    "llama2-70b": "all_gene/llama2-70b_one_pass_gene_merge_0.5.jsonl",
    "llama3.1-8b": "all_gene/llama3.1-8b_one_pass_gene_merge_0.5.jsonl",
    "llama3.2-3b": "all_gene/llama3.2-3b_one_pass_gene_merge_0.5.jsonl",
    "llama3.1-70b": "all_gene/llama3.1-70b_one_pass_gene_merge_0.5.jsonl",
}

analyze_multiple_models(model_paths_dict)
