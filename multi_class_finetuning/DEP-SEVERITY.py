import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from transformers import AutoTokenizer
from torch.optim import AdamW
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
model_root = '/root/autodl-tmp'
base_model = 'Qwen1.5-0.5B-Chat'
df = pd.read_csv('./mental_dataset/DEP-SEVERITY/DEP-SEVERITY-English.csv').dropna(subset=['text', 'label'])

MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 2e-4
manual_seed = 2
loss_fn = nn.CrossEntropyLoss()
best_val_weighted_f1 = 1
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
np.random.seed(manual_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    'minimum': 0,    
    'mild': 1,
    'moderate': 2,
    'severe': 3
}
df['label'] = df['label'].map(label_map)
assert df['label'].isna().sum() == 0, "存在未映射的标签值"
y = df['label'].to_numpy()
x = df['text']
x_train, x_remain, y_train, y_remain = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_remain, y_remain,
    test_size=0.5,
    stratify=y_remain,
    random_state=42
)
val_x = x_val.tolist()
test_x = x_test.tolist()
val_y = np.array(y_val)
test_y = np.array(y_test)

# Random upsampling
train_indices = np.arange(len(x_train)).reshape(-1, 1)
ros = RandomOverSampler(sampling_strategy='all', random_state=42)
X_res, y_res = ros.fit_resample(train_indices, y_train)
x_train_balanced = [x_train.iloc[i] for i in X_res.flatten()]
y_train_balanced = y_res

y_train_balanced = np.array(y_train_balanced)  

wandb.init(
        project="Qwen-mental-tune",
        config={
            "epochs": EPOCHS,
            })

# 准备模型，并测试
model = Qwen_LoRA_4(model_root=model_root,
                  base_model=base_model,
                  lora_rank=4,
                  lora_dropout=0.2).to(device)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root, base_model))
train_loader = build_dataloader(x_train_balanced, y_train_balanced, max_len=MAX_LEN,
                                batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)
val_loader = build_dataloader(val_x, val_y, max_len=MAX_LEN,
                               batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)
test_loader = build_dataloader(test_x, test_y, max_len=MAX_LEN,
                               batch_size=TRAIN_BATCH_SIZE, tokenizer=tokenizer)
n_steps_per_epoch = math.ceil(len(train_loader.dataset) / TRAIN_BATCH_SIZE)
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
predictions = train_val_epoch(EPOCHS, model, train_loader, val_loader,
                              test_loader, optimizer, loss_fn,
                              n_steps_per_epoch, device, scheduler)
wandb.finish()
