
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('results_json_paths', nargs='*')
    args = parser.parse_args()

    all_scores_dict = {}

    for path in args.results_json_paths:
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if k not in all_scores_dict:
                all_scores_dict[k] = []
            all_scores_dict[k].append(v)
    
    mean_scores_dict = {}

    for k, v in all_scores_dict.items():
        mean_scores_dict[k] = np.mean(v)

    if 'test_acc' in mean_scores_dict:
        mean_scores_dict['test_accuracy_best'] = mean_scores_dict['test_acc'] * 100
    if 'test2_acc' in mean_scores_dict:
        mean_scores_dict['test2_accuracy_best'] = mean_scores_dict['test2_acc'] * 100
    if 'train_acc' in mean_scores_dict:
        mean_scores_dict['train_accuracy_best'] = mean_scores_dict['train_acc'] * 100
    if 'val_acc' in mean_scores_dict:
        mean_scores_dict['val_accuracy_best'] = mean_scores_dict['val_acc'] * 100


    results_file = 'results.json'
    with open(results_file, "w") as outfile: 
            json.dump(mean_scores_dict, outfile)

