import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from functions import *

dataset_list = ['DR', 'dreaddit', 'Irf', 'MultiWD']
start_index = [0, 405, 819, 2932]
end_index = [405, 819, 2932, 5373]
generated = {}
golden = {}
path = '/root/gpt_compare/generated_predictions_qwen.jsonl'

all_data = []

# If the generated output is a dictionary per line.
with open(path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        all_data.append(data)
# If the generated result is a list
'''with open(path, 'r', encoding='utf-8') as file:
    all_data = json.load(file)'''


print(len(all_data))
for i, (start, end, name) in enumerate(zip(start_index, end_index, dataset_list)):
    data = all_data[start:end]
    generated[name] = []
    golden[name] = []
    for line in data:
        prompt = line['prompt']
        label = line['label']
        predict = line['predict']
        generated[name].append(predict)
        golden[name].append(label)

score_dict = calculate_f1(generated, golden)
print(score_dict)





