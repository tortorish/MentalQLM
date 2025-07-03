import json
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

def process_item(item):
    client = OpenAI(base_url="http://localhost:5004/v1", api_key="EMPTY")
    
    completion = client.chat.completions.create(
        model="/root/autodl-tmp/mentalqlm",
        messages=[{
            "role": "user",
            "content": item["instruct"]
        }],
        temperature=0.7,
        max_tokens=512
    )
    
    return {
        "instruct": item["instruct"],
        "output": item["output"],
        "generated": completion.choices[0].message.content.strip()
    }

with open('/root/python_prj/MentalLLaMA/my_data/merged_output.json') as f:
    data = json.load(f)

with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    results = list(tqdm(executor.map(process_item, data), 
                      total=len(data), 
                      desc="Processing"))

with open('/root/python_prj/MentalLLaMA/revision/generated_results.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)