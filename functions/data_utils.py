import os
import json
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import random
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer


def generate_response(model, tokenizer, test_data, device, batch_size):
    generated_text = {}
    goldens = {}

    model.to(device)

    for dataset_name in test_data.keys():
        print('Generating for dataset: {}'.format(dataset_name))
        queries, golden = test_data[dataset_name]
        goldens[dataset_name] = golden
        responses = []

        for i in range(0, len(queries), batch_size):
            batch_data = queries[i: min(i+batch_size, len(queries))]
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            #print(inputs)
            #final_input = inputs.input_ids
            input_ids = inputs.input_ids.to(device)  
            attention_mask = inputs.attention_mask.to(device)
            #print(final_input)
            generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=256)  
            for j in range(generate_ids.shape[0]):
                truc_ids = generate_ids[j][len(input_ids[j]) :] 
                response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                responses.append(response)
            print(i)
        generated_text[dataset_name] = responses

    return generated_text, goldens


def calculate_f1(generated, goldens):
    score_dict = {}
    auc_dict = {}
    for dataset_name in generated.keys():
        golden = goldens[dataset_name]
        outputs = generated[dataset_name]

        output_label = []
        golden_label = []
        count = 0


        for ref, output in zip(golden, outputs):
            ref_an = ref.split("Reasoning:")[0]
            output_an = output.split("Reasoning:")[0]

            if dataset_name == 'swmh':
                if 'no mental' in output_an.lower():
                    output_label.append(0)
                elif 'suicide' in output_an.lower():
                    output_label.append(1)
                elif 'depression' in output_an.lower():
                    output_label.append(2)
                elif 'anxiety' in output_an.lower():
                    output_label.append(3)
                elif 'bipolar' in output_an.lower():
                    output_label.append(4)
                else:
                    count += 1
                    output_label.append(0)

                if 'no mental' in ref_an.lower():
                    golden_label.append(0)
                elif 'suicide' in ref_an.lower():
                    golden_label.append(1)
                elif 'depression' in ref_an.lower():
                    golden_label.append(2)
                elif 'anxiety' in ref_an.lower():
                    golden_label.append(3)
                elif 'bipolar' in ref_an.lower():
                    golden_label.append(4)
                else:
                    output_label = output_label[:-1]

            elif dataset_name == 't-sid':
                if 'depression' in output_an.lower():
                    output_label.append(2)
                elif 'suicide' in output_an.lower():
                    output_label.append(1)
                elif 'ptsd' in output_an.lower():
                    output_label.append(3)
                elif 'no mental' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    output_label.append(0)

                if 'depression' in ref_an.lower():
                    golden_label.append(2)
                elif 'suicide or self-harm' in ref_an.lower():
                    golden_label.append(1)
                elif 'ptsd' in ref_an.lower():
                    golden_label.append(3)
                elif 'no mental disorders' in ref_an.lower():
                    golden_label.append(0)

            elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
                if 'yes' in output_an.lower():
                    output_label.append(1)
                elif 'no' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    output_label.append(0)

                if 'yes' in ref_an.lower():
                    golden_label.append(1)
                elif 'no' in ref_an.lower():
                    golden_label.append(0)

            elif dataset_name == 'SAD':
                if 'school' in output_an.lower():
                    output_label.append(0)
                elif 'financial' in output_an.lower():
                    output_label.append(1)
                elif 'family' in output_an.lower():
                    output_label.append(2)
                elif 'social' in output_an.lower():
                    output_label.append(3)
                elif 'work' in output_an.lower():
                    output_label.append(4)
                elif 'health' in output_an.lower():
                    output_label.append(5)
                elif 'emotion' in output_an.lower():
                    output_label.append(6)
                elif 'decision' in output_an.lower():
                    output_label.append(7)
                elif 'other' in output_an.lower():
                    output_label.append(8)
                else:
                    count += 1
                    output_label.append(8)

                if 'school' in ref_an.lower():
                    golden_label.append(0)
                elif 'financial problem' in ref_an.lower():
                    golden_label.append(1)
                elif 'family issues' in ref_an.lower():
                    golden_label.append(2)
                elif 'social relationships' in ref_an.lower():
                    golden_label.append(3)
                elif 'work' in ref_an.lower():
                    golden_label.append(4)
                elif 'health issues' in ref_an.lower():
                    golden_label.append(5)
                elif 'emotion turmoil' in ref_an.lower():
                    golden_label.append(6)
                elif 'everyday decision making' in ref_an.lower():
                    golden_label.append(7)
                elif 'other stress causes' in ref_an.lower():
                    golden_label.append(8)
                else:
                    print(golden.index(ref), ref_an)

            elif dataset_name == 'CAMS':
                if 'none' in output_an.lower():
                    output_label.append(0)
                elif 'bias' in output_an.lower():
                    output_label.append(1)
                elif 'jobs' in output_an.lower():
                    output_label.append(2)
                elif 'medication' in output_an.lower():
                    output_label.append(3)
                elif 'relationship' in output_an.lower():
                    output_label.append(4)
                elif 'alienation' in output_an.lower():
                    output_label.append(5)
                else:
                    count += 1
                    output_label.append(0)

                if 'none' in ref_an.lower():
                    golden_label.append(0)
                elif 'bias or abuse' in ref_an.lower():
                    golden_label.append(1)
                elif 'jobs and career' in ref_an.lower():
                    golden_label.append(2)
                elif 'medication' in ref_an.lower():
                    golden_label.append(3)
                elif 'relationship' in ref_an.lower():
                    golden_label.append(4)
                elif 'alienation' in ref_an.lower():
                    golden_label.append(5)

        avg_accuracy = round(accuracy_score(golden_label, output_label) * 100, 2)
        weighted_f1 = round(f1_score(golden_label, output_label, average='weighted') * 100, 2)
        micro_f1 = round(f1_score(golden_label, output_label, average='micro') * 100, 2)
        macro_f1 = round(f1_score(golden_label, output_label, average='macro') * 100, 2)
        # recall = round(recall_score(f_labels, outputs, average='weighted')*100, 2)
        score_dict[dataset_name] = {'avg_accuracy': avg_accuracy, 'weighted_f1': weighted_f1,
                                    'micro_f1': micro_f1, 'macro_f1': macro_f1}


    return score_dict


def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        summary = json.load(file)
    return summary


def write_json(content, path):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=4)


def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length, device):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')


def get_label_SAD(output_an):
    if 'school' in output_an.lower():
        return 0
    elif 'financial problem' in output_an.lower():
        return 1
    elif 'family issues' in output_an.lower():
        return 2
    elif 'social relationships' in output_an.lower():
        return 3
    elif 'work' in output_an.lower():
        return 4
    elif 'health issues' in output_an.lower():
        return 5
    elif 'emotion turmoil' in output_an.lower():
        return 6
    elif 'everyday decision making' in output_an.lower():
        return 7
    elif 'other stress causes' in output_an.lower():
        return 8
    else:
        return 10


def get_label_binary(output_an):
    if 'yes' in output_an.lower():
        return 1
    elif 'no' in output_an.lower():
        return 0


def get_dataset(path, dataset_name='SAD', exclude_list=None):
    old_list = read_json(path)
    if exclude_list is not None:
        old_list = [item for idx, item in enumerate(old_list) if idx not in exclude_list]
    new_list = []
    output_label_list = []
    for item in old_list:
        input = item['instruct']
        output = item['output']
        if dataset_name == 'SAD':
            output_label = get_label_SAD(output)
        else:
            output_label = get_label_binary(output)
        start_quote = input.find('"') + 1 
        end_quote = input.rfind('"')
        extracted_text = input[start_quote:end_quote]
        new_list.append(extracted_text)
        output_label_list.append(output_label)
    '''for item in new_list:
        if 'Consider' in item or 'Question' in item:
            print('false', new_list.index(item))'''
    return new_list, np.array(output_label_list)


def filter_samples(train_x, train_y):
    count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    filtered_train_x = []
    filtered_train_y = []

    for x, y in zip(train_x, train_y):
        if y in [0, 1, 2, 3, 4, 5, 6, 7]:
            if count_dict[y] < 173:
                filtered_train_x.append(x)
                filtered_train_y.append(y)
                count_dict[y] += 1
        else:
            filtered_train_x.append(x)
            filtered_train_y.append(y)

    return filtered_train_x, filtered_train_y


def k_center_greedy(train_x, k):
    """
    使用KCenterGreedy算法选择中心点。

    :param train_x: 形状为 (n_samples, n_features) 的numpy数组，包含所有样本。
    :param k: 期望选择的中心点数量。
    :return: 选择的中心点的索引。
    """
    # 计算样本间的距离矩阵
    np.random.seed(42)
    distance_matrix = pairwise_distances(train_x)
    # 初始化中心点集合
    centers = []
    # 随机选择第一个中心点
    center_id = np.random.randint(len(train_x))
    centers.append(center_id)

    # 使用贪心策略选择剩余的中心点
    for _ in tqdm(range(k - 1), desc="Selecting Centers"):
        # 计算每个样本到最近中心点的距离
        closest_center_distance = np.min(pairwise_distances(train_x, train_x[centers]), axis=1)
        # 选择离现有中心点最远的样本作为新的中心点
        new_center_id = np.argmax(closest_center_distance)
        centers.append(new_center_id)

    return centers


def qwen_tokenizer(input_list, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    length = [(tokenizer(item, return_tensors='pt')['input_ids'].shape[-1], input_list.index(item)) for item in input_list]
    return length


def filter_by_perplexity(input_list, is_none=False):
    if is_none:
        ppl_100_list = [p for p in input_list if p < 100]
    else:
        ppl_100_list = input_list
    Q1 = np.percentile(ppl_100_list, 25)
    Q3 = np.percentile(ppl_100_list, 75)
    IQR = Q3 - Q1

    # 定义异常值的范围
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((input_list < lower_bound) | (input_list > upper_bound))

    return outlier_indices[0]


def write_to_file(train_x, train_y, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(len(train_x)):
            # 确保标签是一个字符串，并且字符串和标签之间用'\t'分隔
            file.write(f"{train_x[i]}\t{train_y[i]}\n")


def get_dataset_random(path, dataset_name='SAD', proportion=0.5, exclude_list=None):
    old_list = read_json(path)
    if exclude_list is not None:
        old_list = [item for idx, item in enumerate(old_list) if idx not in exclude_list]
    new_list = []
    output_label_list = []
    for item in old_list:
        input = item['instruct']
        output = item['output']
        if dataset_name == 'SAD':
            output_label = get_label_SAD(output)
        else:
            output_label = get_label_binary(output)
        start_quote = input.find('"') + 1  # 加1是为了跳过引号本身
        end_quote = input.rfind('"')
        # 提取引号之间的文本
        extracted_text = input[start_quote:end_quote]
        new_list.append(extracted_text)
        output_label_list.append(output_label)
    for item in new_list:
        if 'Consider' in item or 'Question' in item:
            print('false', new_list.index(item))
    random.seed(42)
    total_points = len(new_list)
    num_to_select = total_points * proportion
    indices = list(range(total_points))
    selected_indices = random.sample(indices, num_to_select)
    new_list = [new_list[i] for i in selected_indices]
    output_label_list = [output_label_list[i] for i in selected_indices]

    return new_list, np.array(output_label_list)


def k_center_cuda(X, k, device='cuda'):
    """
    K-Center Greedy Algorithm with CUDA acceleration and tqdm progress bar.

    Parameters:
    - X: Tensor of shape (a, b) where 'a' is the number of samples and 'b' is the embedding dimension.
    - k: The number of centers to select.
    - device: Device to run computations on ('cuda' for GPU or 'cpu' for CPU).

    Returns:
    - centers: Indices of the selected centers.
    """
    # Move data to the specified device
    X = torch.tensor(X).to(device)

    # Initialize the first center randomly
    n_samples = X.shape[0]
    centers = [torch.randint(0, n_samples, (1,), device=device)]

    # Compute initial distances from the first center to all other points
    distances = torch.cdist(X[centers[0]].unsqueeze(0), X).squeeze()

    # Use tqdm to create a progress bar
    for _ in tqdm(range(1, k), desc="Selecting centers", total=k - 1):
        # Find the point with the maximum distance to its nearest center
        new_center = torch.argmax(distances)

        # Add this point as a new center
        centers.append(new_center)
        temp = X[new_center].unsqueeze(0)
        # Update the distances to the nearest center
        new_distances = torch.cdist(X[new_center].unsqueeze(0), X).squeeze()
        distances = torch.min(distances, new_distances)

    return torch.tensor(centers, device='cpu')


def read_json_dict(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的数据
            record = json.loads(line)
            # 将解析后的字典添加到列表中
            data_list.append(record)
    return data_list


def convert_labels(predictions, path):
    new_list = []
    old_list = read_json(path)
    assert len(old_list) == len(predictions)
    for i in range(len(predictions)):
        input = old_list[i]['instruct']
        output = old_list[i]['output']
        reasoning = output.split("Reasoning:")[1]
        prediction = predictions[i]
        if prediction == 0:
            label = 'school'
        elif prediction == 1:
            label = 'financial problem'
        elif prediction == 2:
            label = 'family issues'
        elif prediction == 3:
            label = 'social relationships'
        elif prediction == 4:
            label = 'work'
        elif prediction == 5:
            label = 'health issues'
        elif prediction == 6:
            label = 'emotion turmoil'
        elif prediction == 7:
            label = 'everyday decision making'
        elif prediction == 8:
            label = 'other stress causes'

        post = input.split("Question:")[0]
        question = ("Question: This post shows the stress cause related to {}, "
                    "explain the reasoning of it step by step").format(label)
        data_dict = {"instruct": post + question, "output": reasoning}
        new_list.append(data_dict)
    json_data = json.dumps(new_list, indent=4)
    with open("predictions.json", "w") as json_file:
        json_file.write(json_data)
    print("Prediction has been saved, Use LLM to generate the reasoning for the label in the next step")



