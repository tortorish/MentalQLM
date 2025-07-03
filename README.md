# Introduction

Code implementation for "MentalQLM: A lightweight large language model for mental healthcare based on instruction tuning and dual LoRA modules"

Mental disorders pose significant challenges to healthcare systems and have profound social implications. The rapid development of large language model (LLM) presents new opportunities for improving mental healthcare. However, existing approaches primarily rely on instruction tuning and few-shot in-context learning with massive datasets and large-scale backbone models, leading to significant computational costs. 
To address these limitations, we propose MentalQLM, a novel lightweight LLM that leverages a dual Low-Rank Adaptation (LoRA) strategy for parameter-efficient fine-tuning.
The development of our proposed MentalQLM includes two key stages. Firstly, we perform dataset pruning based on perplexity and diversity analysis to reduce computational load. The first LoRA module is applied during instruction tuning to adapt the base LLM for mental health classification. Secondly, we introduce a dense layer augmented with a second LoRA module, fine-tuned specifically to boost performance on complex multi-class classification problems.
%
Experimental results demonstrate that our proposed MentalQLM, with only 0.5 billion parameters, achieves an average weighted F1-score of 0.778 on mental disorder diagnosis across five benchmark datasets. It outperforms the state-of-the-art instruction-tuned MentaLLaMA-Chat-13B model by 3.2\%, and the few-shot tuned GPT-4 model by 17.7\%. This promising performance, combined with its significantly lower resource requirements, positions our developed MentalQLM as a cost-effective and efficient solution for real-world mental healthcare applications, especially in computationally constrained environments.


We build our project based on [The MentaLLaMA Project](https://github.com/SteveKGYang/MentalLLaMA), [The Cherry_LLM project](https://github.com/tianyi-lab/Cherry_LLM), and [The LLaMA-Factory project](https://github.com/hiyouga/LLaMA-Factory). 
We sincerely appreciate the community for their contribution!

# File Structure
```bash
├───data_selection
│   ├───get_perplexity.py
│   └───kcenter_selelct.py
├───evaluate
│   ├───LLM_accelerated
│   ├───F1-Score.py
│   └───GPT-Score.py
├───functions
│   ├───__init__.py
│   ├───data_utils.py
│   ├───model.py
│   └───train_utils.py
├───Instruction tuning
│   ├───__init__.py
│   └───functions.py
├───mental_dataset
│   ├───DEP-SEVERITY
│   └───IMHI
├───multi-class finetuning
│   ├───DEP-SEVERITY.py
│   ├───SAD.py
├───perplexity
│   ├───DR.pt
│   ├───dreaddit.pt
│   ├───Irf.pt
│   ├───MultiWD.pt
│   ├───SAD.pt
├───data_process.ipynb
```

- The `data_selection/` folder contains the code for data selection based on quality and diversity.
- The `evaluate/` folder contains the script to evaluate the generated result in terms of F1-Score and GPT-Score.
- The `functions/` folder contains the script for dataset buildinng, model construction and training.
- The `Instruction tuning/` folder contains the yaml script， which can be used to train with LLaMA-Factory.
- The `mental_dataset/` folder contains two datasets. The first is the DEPSEVERIT dataset and its six translated version in English, Turkish, French, Portuguese, German, Greek and Finnish
- The `multi-class finetuning/` folder contain the folder to finetune the LLM using LoRA based on the DEP-SEVERITY dataset and SAD dataset.
- The `perplexity/` folder contain perplexity distribution of the IMHI dataset, which can be used for data quality pruning.

# Running Guide

To train this model, you need to construct the dataset at first.
Run the `data_process.ipynb` notebook and get the training set and test set.

## Training Models

### Data selection
```bash
cd [parent folder of the project]//data_selection//
# Firstly, get the perplexity distribution of a subdataset
python get_perplexity.py
# We also prepare the perplexity file in the 'Instruction/' folder. Thus you can use them directly.
# The next step is to maximize data coverage by using the k-center-greedy algorithm
python kcenter_select.py
```

### Instruction tuning for the binary classification datasets
After you get the selected dataset, the next step is to instruction tune the LLM. We recommend using the LLaMA-Factory training framework.
To retrain the models, you can run `train_model.yaml` in the `Instruction/` folder based on the LLaMA-Factory training framework.
```bash
cd [parent folder of the project]/instruction_tuning/
llamafactory-cli train train_mentalqlm.yaml
```

### LoRA enabled finetuning for multi-class classification datasets

```bash
cd [parent folder of the project]/multi-class finetuning/
# finetune the LLM on the DEP-SEVERITY dataset
python .//dep-severity.py
# finetune the LLM on the SAD dataset
python .//SAD.py
```

## Evaluating Models

For reproducibility, we recommend using the LLaMA-Factory framework to generate the answer, which can produce ROUGE and BLEU score.
We also provide the script `F1-Score.py` to calculate the F1-score and the `GPT-Score.py` to calculate the GPT-score.
```bash
cd [parent folder of the project]/evaluate/
# First you can use the LLaMAFactory to generate the output, which can output the ROUGE and BLEU score.
llamafactory-cli train evaluate.yaml
# Based on the prediction result generated by LLaMAFactory, you can get the F1-Score by running the following script.
python .//F1-Score.py
# get the GPT-Score of the generated results. Note you need to get the result of two different models.
python .//GPT-Score.py

# you can also use the VLLM or SGlang accerleration framework to get the result
# first run the model on a local port
bash vllm_qwen.sh
# Then run the script to get the final result
python infer_vllm_thread.py
```



