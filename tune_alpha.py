from attack_util import rad_attack_debug, get_data, get_model_and_processor, mnist, llava_id
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def tune_alpha(model, processor, mnist, data_range):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    results = pd.DataFrame(columns=['alpha', 'distance', 'iters'])
    for alpha in np.logspace(-3, 1, 10):
        print("alpha =", alpha)
        distances = []
        iterations = []
        for i in tqdm(data_range):
            inputs, label_id = get_data(mnist[i], processor)
            _, distance, iters = rad_attack_debug(model, processor, inputs, label_id, alpha=alpha, debug=False)
            distances.append(distance)
            iterations.append(iters)
        results = pd.concat([results, pd.DataFrame({'alpha': alpha, 'distance': distances, 'iters': iterations})])
    return results

if __name__ == '__main__':
    results_dir = 'results'
    model, processor = get_model_and_processor(llava_id)
    data_range = range(800)
    results = tune_alpha(model, processor, mnist, data_range)
    for alpha, df in results.group_by('alpha'):
        print(df.describe())
    results.to_csv(f'{results_dir}/tune_alpha.py')
