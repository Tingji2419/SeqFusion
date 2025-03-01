import pandas as pd
import matplotlib.pyplot as plt


# Function to parse the filename and extract dataset and model
def parse_name(name):
    parts = name.split('_')

    model = parts[0]
    target_dataset = parts[2]
    pred_len = parts[5]

    return model, target_dataset, pred_len


def find_mean_mse_results(filename, rank_key='mse'):
    from collections import defaultdict
    results = defaultdict(list)
    with open(filename, 'r') as file:
        while True:
            name_line = file.readline().strip()
            if not name_line:
                break  # End of file
            data_line = file.readline().strip()
            file.readline()  # Skip the blank line

            source_model, target_dataset, pred_len = parse_name(name_line)
            
            all_values = {i.strip().split(':')[0]: float(i.strip().split(':')[1]) for i in data_line.split(',')}
            rank_value = all_values[rank_key]# locals()[rank_key]
            key = (source_model, target_dataset)
            results[key].append(rank_value)
    for key, value in results.items():
        mean_value = sum(value) / len(value)
        print(f"Mean rank_key: {mean_value:.4f}; {key}")


    data = {}
    for model_dataset, mse_value in results.items():
        model, dataset = model_dataset  # Split the key
        if '-' in dataset:
            dataset = dataset.split('-')[0]
        if model not in data:
            data[model] = {}
        data[model][dataset] = sum(mse_value) / len(mse_value)
    
    datasets = sorted({ds for md in data.keys() for ds in data[md].keys()})
    for model, model_data in data.items():
        mse_values = list(model_data.values())
        mean_mse = sum(mse_values) / len(mse_values)
        data[model]['MEAN'] = mean_mse
    for dataset in datasets:
        print(dataset, end='\t')
    print('\n')
    # Print the rows for each model
    for model, model_data in sorted(data.items()):
        print(model, end='\t')
        for dataset in datasets:
            # Print MSE value if it exists, otherwise print a placeholder
            mse_value = model_data.get(dataset, "")
            print(f"{mse_value:.4f}" if isinstance(mse_value, float) else "", end='\t')
        print(f"{model_data['MEAN']:.4f}")


import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--f', type=str, default='benchmark1.txt')

args = parse.parse_args()
rank_key = 'mse'

find_mean_mse_results(args.f, rank_key)



