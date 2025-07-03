# This script is used to compute the perplexity of the reasoning sentence
import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn

log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Get perplexity distribution of the five IMHI subdatasets sequentially.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='.//mental_dataset//mental_istct_train_val_DR.json')
    parser.add_argument("--save_path", type=str, default='DR.pt')
    parser.add_argument("--model_name_or_path", type=str, default='/root/autodl-tmp/Qwen1.5-0.5B-Chat')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    return args


def get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, label_text, max_length):
    input_ids = tokenizer.encode(whole_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    start_index = whole_text.rfind(label_text)
    start_id = len(tokenizer.encode(whole_text[:start_index]))
    end_token = input_ids.shape[1]
    label_id_len = len(tokenizer.encode(label_text)) - 1


    labels = input_ids.clone()
    labels[0, :start_id] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i - 1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    losses = losses[start_id-1:start_id+label_id_len-1]

    return perplexity.to('cpu'), losses


def main():
    args = parse_args()
    print(args)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', cache_dir='../cache',
                                                 output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    model.eval()

    if args.save_path[-3:] != '.pt':
        args.save_path += '.pt'
    if os.path.exists(args.save_path):
        print('save_path exists!')
        raise Exception

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    import time
    strat_time = time.time()
    new_data = []
    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['instruct']
        output_i = data_i['output']
        output_label = output_i.split("Reasoning:")[0]
        whole_text = instruct_i + output_i

        temp_data_i = {}


        perplexity, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer,
                                                                                 model,
                                                                                 whole_text,
                                                                                 output_label,
                                                                                 args.max_length)
        temp_data_i['token_loss'] = [loss_list_condition]
        temp_data_i['perplexity'] = perplexity

        new_data.append(temp_data_i)
        pass

    print('New data len:', len(new_data))
    torch.save(new_data, args.save_path)

    print('Time Used:', (time.time() - strat_time) / 60, '(min)')


if __name__ == "__main__":
    main()
