from attack_util import attack3, attack4, get_mnist_instance, get_model_and_processor, get_target, mnist, llava_id
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

def tune_alpha(model, processor, mnist, data_range, id):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    results = pd.DataFrame(columns=['alpha', 'distance', 'iters'])
    for alpha in np.logspace(-1, 4, 12):
        print("alpha =", alpha)
        distances = []
        iterations = []
        for i in tqdm(data_range):
            inputs, label_id = get_mnist_instance(mnist[i], processor)
            if id == 0:
                target_id = get_target(model, processor, inputs, label_id)
                _, distance, iters = attack3(model, inputs, target_id, alpha=alpha, debug=False)
            if id == 1:
                _, distance, iters = attack4(model, inputs, label_id, alpha=alpha, debug=False)
            distances.append(distance)
            iterations.append(iters)
        results = pd.concat([results, pd.DataFrame({'alpha': alpha, 'distance': distances, 'iters': iterations})])
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n', type=int, default=600)
    args = parser.parse_args()
 
    if args.id == 0:
        print("Attack 3 Finetuning")
    elif args.id ==1:
        print("Attack 4 Finetuning")
    else:
        exit("invalid attack id")
    results_dir = 'results'
    model, processor = get_model_and_processor(llava_id)
    data_range = range(args.n)
    results = tune_alpha(model, processor, mnist, data_range, args.id)
    for alpha, df in results.groupby('alpha'):
        print("alpha:",alpha)
        print("success rate:", len(df.dropna())/len(df))
        print(df.describe())
        print("\n")
    results.to_csv(f'{results_dir}/tune_alpha.csv')
