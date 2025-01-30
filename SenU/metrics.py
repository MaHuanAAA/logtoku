import numpy as np
from scipy.special import softmax, digamma

def topk(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices

def get_eu(mode="prob", k=None):
    if mode == "eu":
        if k is None:
            raise ValueError("k must be provided for 'eu' mode.")

        def eu(logits):
            top_k = k
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
            return mean_scores

        return eu

    elif mode == "prob":
        def eu(logits):
            logits = softmax(logits)
            top_k = 1
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            return top_values[0]

        return eu

    elif mode == "entropy":
        def eu(logits):
            probs = softmax(logits)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy

        return eu

    elif mode == "au":
        def cal_au(logits):
            top_k = k
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values = np.partition(logits, -top_k)[-top_k:]
            alpha = np.array([top_values])
            alpha_0 = alpha.sum(axis=1, keepdims=True)
            psi_alpha_k_plus_1 = digamma(alpha + 1)
            psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
            result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
            return result.sum(axis=1)[0]

        return cal_au

    elif mode == "eu_2":
        def eu(logits):
            top_k = 2
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
            return mean_scores

        return eu

    elif mode == "au_2":
        def cal_au(logits):
            top_k = 2
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values = np.partition(logits, -top_k)[-top_k:]
            alpha = np.array([top_values])
            alpha_0 = alpha.sum(axis=1, keepdims=True)
            psi_alpha_k_plus_1 = digamma(alpha + 1)
            psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
            result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
            return result.sum(axis=1)[0]

        return cal_au

    else:
        raise ValueError(f"Unsupported mode: {mode}")

