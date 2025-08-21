# Estimating LLM Uncertainty with Evidence & Improving Semantic Entropy

This repository contains the official implementation for the paper **"Estimating LLM Uncertainty with Evidence (Logits)"**, introducing a novel approach for efficient uncertainty quantification in large language models. Additionally, it includes **how to utilize this method to enhance Semantic Entropy**. The relevant technical report is currently being compiled, and the method can be found in [Semantic Energy](https://github.com/MaHuanAAA/SemanticEnergy).

## Citation

```
@article{ma2025estimating,
  title={Estimating LLM Uncertainty with Evidence},
  author={Ma, Huan and Chen, Jingdong and Joey Tianyi Zhou and Wang, Guangyu and Zhang, Changqing},
  journal={arXiv preprint arXiv:2502.00290},
  year={2025}
}
```

## FAQs

***Q1: About the value ranges of AU and EU***

The values of AU are not within the [0,1]. Theoretically, the range of AU can be (0,+∞), while the value of EU is between [0,1]. However, in practice, the ranges of AU or EU for different tokens are very close. Currently, AU and EU mainly serve as indicators of rank, and their specific scale is a question worth studying. Of course, this does not affect the experiments. Similar to other comparative methods like entropy, we can currently select thresholds based on rank, and adopting AUROC as the comparative metric can eliminate the impact of manually selecting thresholds.

***Q2: QA Performance***

Some have reported that the performance on TriviaQA is relatively poor in the hallucination detection task, and our tests have confirmed this issue. The cause of this problem lies in the fact that LogTokU only deems a response unreliable when both EU and AU are high. However, in some semantically sensitive QA scenarios, certain responses may exhibit semantic conflicts. For instance, in sentiment analysis, a next-token might receive a high score for both "positive" and "negative" sentiments simultaneously. Although this suggests that the model has encountered many similar scenarios, it still fails to make an accurate judgment. This issue has been resolved in the improved version. Please refer to the [Semantic Energy](https://github.com/MaHuanAAA/SemanticEnergy) for details.


## Requirements

```
conda env create -f env.yml
conda activate logtoku
```

## Preparation for models and datasets

### Model Preparation

Here are all the models required for our experiment. 

```bash
- meta-llama/Llama-2-{7b,13b,70b}-chat-hf
- meta-llama/Llama-3-{3B,8B,70B}-Instruct
- microsoft/deberta-v2-xlarge-mnli
- lucadiliello/BLEURT-20
```

You can directly create the folders and run `download_model.py` to download them to the specified directories (please make sure your Hugging Face access token is properly configured).

#### Download Instructions

1. Configure your Hugging Face access token:
   ```bash
   huggingface-cli login
   ```

2. Download all models:

   ```bash
   mkdir models
   python download_model.py 
   ```

### Dataset Preparation

The dataset used in these experiments can be downloaded automatically when you run the following workflow.

## Experiment 1: LogTokU-guided Decoding

### Directory Structure

This sub-experiment is located in the `DynDecoding` directory. For clarity, we present the Directory structure as follows:

```bash
DynDecoding/
├── configs/
│   └── prompts.json
├── outputs/
│   └── llama2_7b.jsonl
│   └── more...
├── gene.py # 
├── eval.ipynb # Evaluating the scores of different decoding strategies
```

### Experiment Workflow

The workflow to reproduce the results is very clear and simple. 

1. You only need to run `gene.py` to generate intermediate files like `llama2-7b.jsonl` in the `output` folder. 

   ```bash
   python gene.py --exp llama2_7b --gpuid 0 --quantize False
   ```
2. Then, you can evaluate the scores of different decoding strategies in our `eval.ipynb` like this: (`Llama2-7b`)
   ```bash
   Model Name      Score      Score Rate (%) 
   ----------------------------------------
   Greedy decoding 2525       77.48%
   Top-2 Sampling  2520       77.32%
   prob            2558       78.49%
   entropy         2585       79.32%
   LogTokU         2831       86.87%
   ```

## Experiment 2: LogTokU-guided Response Uncertainty Estimation

### Directory Structure

This sub-experiment is located in the `SenU` directory. For clarity, we present the file structure as follows:

```bash
SenU/
├── muti_eval/
│   └── llama2_chat_7B_muti_pass_gene_0.5_metrics.jsonl
│   └── llama2_chat_7B_muti_pass_gene_0.5_final.jsonl
│   └── more...
├── all_gene/
│   └── llama2_chat_7B_eval_0.5.jsonl
│   └── llama2_chat_7B_muti_pass_gene_0.5.jsonl
│   └── llama2_chat_7B_one_pass_gene_0.5.jsonl
│   └── llama2_chat_7B_one_pass_gene_merge_0.5.jsonl
│   └── more...
├── generate.py  # generate predictions for one-pass and mult-pass methods
├── one_eval.py  # evaluate one-pass prediction correctness and uncertainty
├── muti_eval.py # evaluate multi-pass prediction uncertainty
├── metrics.py   # uncertainty metrics for one-pass methods
└── utils.py  
├── run_eval.sh 
├── run_generate.sh
```

### Experiment Workflow

As outlined in the file structure, our experiment involves generating and evaluating both one-pass and multi-pass outputs. A suggested execution order is as follows:

1. **Generate one-pass predictions**
   
   ```bash
   python generate.py --model_name llama2_chat_7B --gene 1 --mode one_pass --gpuid 0 
   ```
   
2. **Generate multi-pass predictions**
   
   ```bash
   python generate.py --model_name llama2_chat_7B --gene 1 --mode muti_pass --temp 0.5 --gpuid 0
   ```
   
3. **Generate correctness evaluation**
   
   ```bash
   python generate.py --model_name llama2_chat_7B --generate_gt 1 --gpuid 0
   ```
   
4. **Perform uncertainty evaluation for one-pass method**
   
   ```bash
   python one_eval.py  # just set the dict in the code correctly and you can get all metrics at once
   ```
   
5. **Perform uncertainty evaluation for multi-pass method**
   
   ```bash
   python muti_eval.py --model 7b --gpuid 0
   ```

Of course, you can directly run the `run_eval.sh` and `run_generate.sh` scripts we have organized to reproduce the entire experiment.



[ **Poster** | [点击跳转中文海报](./poster.png) ] [ [PPT](./uncertaintyofLLM_EN.pptx) | [中文PPT材料](./UncertaintyPPT.pptx) ]
![Poster](./poster_en.png)




