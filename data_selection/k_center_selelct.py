import os
from tqdm import tqdm
import torch
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from functions import *

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def all_elements_unique(lst):
    if len(lst) == len(set(lst)):
        return True
    else:
        return len(lst) - len(set(lst))


def find_duplicates(lst):
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates


model_path = '//root//autodl-tmp'
model_name = 'Qwen1.5-0.5B-Chat'
dataset_name = ['DR', 'dreaddit', 'Irf', 'MultiWD', 'SAD']
p = 0.75 # change this parameter to get the disred dataset size.
use_k_center = True
Filter_By_Perplexity = True
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, model_name))
model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, model_name), device_map='auto', cache_dir='../cache', output_hidden_states=True)

final_selected_list = []
for i in dataset_name:
    path_train = '//root//python_prj//Mental_LLaMA_release//mental_dataset//mental_istct_train_' + str(i) + '.json'
    path_val = '//root//python_prj//Mental_LLaMA_release//mental_dataset//mental_istct_val_' + str(i) + '.json'
    content_train = read_json(path_train)
    content_val = read_json(path_val)
    content_train.extend(content_val) 

    proportion = int((len(content_train)) * p)
    train_embedding = []

    if Filter_By_Perplexity:
        path_ppl = '//root//python_prj//Mental_LLaMA_release//perplexity//' + str(i) + '.pt'
        pt_ppl = torch.load(path_ppl, map_location=torch.device('cpu'))
        if len(pt_ppl) != len(content_train):
            print("please check size of the dataset")
            sys.exit()
        perplexity_list = [np.array(pt_ppl[i]['perplexity']) for i in range(len(pt_ppl))]
        outlier_indices = [index for index, value in enumerate(perplexity_list) if (np.isnan(value))]
        outlier_indices.extend(filter_by_perplexity(perplexity_list))
        print('duplicate:{}, dataset:{}, original_length:{}, outliers_num:{}, remain:{}'.format(find_duplicates(outlier_indices), i, len(content_train), len(outlier_indices), len(content_train)-len(outlier_indices)))

    if use_k_center:
        if i == 'SAD':
            dataset_name = i
        else:
            dataset_name = "binary"

        # First we need to get the embedding. The prompt is removed and only the post is retained.
        train_x, train_y = get_dataset(path_train, dataset_name, outlier_indices)

        for j in tqdm(range(len(train_x))):
            data_j = train_x[j]
            _, emb_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, data_j, 512, device='cuda')
            train_embedding.append(emb_ins_alone)

        train_embedding = torch.cat(train_embedding, dim=0)
        train_embedding = train_embedding.numpy()
        desired_centers = proportion
        selected_indices = k_center_cuda(train_embedding, desired_centers)
        print('Selected Indices Unique: {}'.format(all_elements_unique(selected_indices)))
        selected_content = [content_train[i] for i in selected_indices]
        final_selected_list.extend(selected_content)


print("length:{}".format(len(final_selected_list)))
write_json(final_selected_list, 'mental_selected_list_' + str(len(final_selected_list)) + '.json')
