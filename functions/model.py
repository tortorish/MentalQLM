import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BertModel, BertTokenizer
import os
from torch import nn
from peft import LoraConfig, get_peft_model


class Qwen_LoRA(torch.nn.Module):

    def __init__(self, model_root, base_model, lora_rank, lora_dropout):
        super(Qwen_LoRA, self).__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(os.path.join(model_root, base_model),
                                                         device_map="auto",
                                                         output_hidden_states=True)
        self.ln = nn.LayerNorm(1024)
        # 增加隐藏层
        self.fc1 = nn.Linear(1024, 9)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2*lora_rank,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.qwen_lora = get_peft_model(self.qwen, self.lora_config)
        self.qwen_lora.print_trainable_parameters()

    def forward(self, input_ids):
        outputs = self.qwen_lora(input_ids)  
        outputs1 = outputs.hidden_states[-1]
        outputs = torch.mean(outputs1, dim=1, keepdim=False)
        outputs = self.ln(outputs)
        outputs = self.fc1(outputs)

        return outputs


class Qwen_LoRA_4(torch.nn.Module):

    def __init__(self, model_root, base_model, lora_rank, lora_dropout):
        super(Qwen_LoRA_4, self).__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(os.path.join(model_root, base_model),
                                                         device_map="auto",
                                                         output_hidden_states=True)
        self.ln = nn.LayerNorm(1024)
        # 增加隐藏层
        self.fc1 = nn.Linear(1024, 4)
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2*lora_rank,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.qwen_lora = get_peft_model(self.qwen, self.lora_config)
        self.qwen_lora.print_trainable_parameters()

    def forward(self, input_ids):
        outputs = self.qwen_lora(input_ids)
        outputs1 = outputs.hidden_states[-1]
        outputs = torch.mean(outputs1, dim=1, keepdim=False)
        outputs = self.ln(outputs)
        outputs = self.fc1(outputs)

        return outputs