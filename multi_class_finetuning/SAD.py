import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from torch.optim import AdamW
from imblearn.over_sampling import RandomOverSampler
from torch import nn
import os
import torch
import wandb
import math
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from functions import *


# Please change the follow parameters according to your setting
model_root = '//root//autodl-tmp'
base_model = 'Qwen1.5-0.5B-Chat'
dataset_name = 'SAD'
dataset_type = 'SAD'
path_train = '//root//python_prj//Mental_LLaMA_release//mental_dataset//mental_istct_train_'+str(dataset_name)+'.json'
path_val = '//root//python_prj//Mental_LLaMA_release//mental_dataset//mental_istct_val_'+str(dataset_name)+'.json'
path_test = '//root//python_prj//Mental_LLaMA_release//mental_dataset//mental_istct_test_'+str(dataset_name)+'.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 32
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-5
manual_seed = 0

loss_fn = nn.CrossEntropyLoss()
best_val_weighted_f1 = 0
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
np.random.seed(manual_seed)

train_x, train_y = get_dataset(path_train, dataset_name=dataset_type, exclude_list=None)
val_x, val_y = get_dataset(path_val, dataset_name=dataset_type)
test_x, test_y = get_dataset(path_test, dataset_name=dataset_type)

ros = RandomOverSampler(sampling_strategy='all', random_state=42)
X_resampled_indices, y_resampled = ros.fit_resample(np.arange(len(train_x)).reshape(-1, 1), train_y)

train_x = [train_x[idx[0]] for idx in X_resampled_indices]
train_y = y_resampled

wandb.init(
        project="Qwen-mental-tune",
        config={
            "epochs": EPOCHS,
            })

model = Qwen_LoRA(model_root=model_root,
                  base_model=base_model,
                  lora_rank=4,
                  lora_dropout=0.2).to(device)

tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root, base_model))
train_loader = build_dataloader(train_x, train_y, max_len=MAX_LEN,
                                batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)

val_loader = build_dataloader(val_x, val_y, max_len=MAX_LEN,
                               batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)

test_loader = build_dataloader(test_x, test_y, max_len=MAX_LEN,
                               batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)
n_steps_per_epoch = math.ceil(len(train_loader.dataset) / TRAIN_BATCH_SIZE)
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
predictions = train_val_epoch(EPOCHS, model, train_loader, val_loader,
                              test_loader, optimizer, loss_fn,
                              n_steps_per_epoch, device, scheduler)
