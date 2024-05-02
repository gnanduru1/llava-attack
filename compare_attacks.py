from attack_util import seed_everything
seed_everything()

import os, sys
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

from tqdm import tqdm
import torch
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import pandas as pd
import argparse
from attack_util import get_model_and_processor, save_img, get_mnist_instance, get_target, attack1, attack2, attack3, attack4, mnist, llava_id

model, processor = get_model_and_processor(llava_id)
def debug_example(i, alpha):
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    # save_img(inputs['pixel_values'], inputs['pixel_values'], f'image_{i}', #results_dir)

    target_id = get_target(model, processor, inputs, label_id)
    print(f"Attack 1&3 target = {processor.decode(target_id)}")

    result1 = attack1(model, inputs, target_id, debug=False)
    if result1 == (None, None, None):
        print("Attack 1 failed")
    _, distance_1, iters_1 = result1
    print(f"Attack: 1 | iters: {iters_1} | distance: {distance_1}")

    # del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    result2 = attack2(model, inputs, label_id, debug=False)
    if result2 == (None, None, None):
        print("Attack 2 failed")
    _, distance_2, iters_2 = result2
    print(f"Attack: 2 | iters: {iters_2} | distance: {distance_2}")

    # del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    result3 = attack3(model, inputs, target_id, alpha=alpha, debug=False)
    if result3 == (None, None, None):
        print("Attack 3 failed")
    _, distance_3, iters_3 = result3
    print(f"Attack: 3 | iters: {iters_3} | distance: {distance_3}")

    # del inputs
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    result4 = attack4(model, inputs, label_id, alpha=alpha, debug=False)
    if result4 == (None, None, None):
        print("Attack 4 failed")
    new_img, distance_4, iters_4 = result4
    print(f"Attack: 4 | iters: {iters_4} | distance: {distance_4}")

    # save_img(new_img, new_img, f'perturbed_image_{i}', results_dir)


def run_and_compare_attacks(model, mnist, id, alpha=100, data_range=range(3000)):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    
    if id==0:
        print("Attack 1")
    elif id==1:
        print("Attack 2")
    elif id==2:
        print("Attack 3")
    elif id==3:
        print("Attack 4")
    else:
        print(f"Invalid id: {id}")
        return
    
    # results = pd.DataFrame(columns=[f'attack {id} distance', f'attack {id} iters'])
    results = pd.DataFrame(columns=['distance', 'iters'])
    for i in tqdm(data_range):
        row = mnist[i]
        inputs, label_id = get_mnist_instance(row, processor)
        
        if id==0:
            target_id = get_target(model, processor, inputs, label_id)
            _, distance, iters = attack1(model, inputs, target_id)
        elif id==1:
            _, distance, iters = attack2(model, inputs, label_id)
        elif id==2:
            target_id = get_target(model, processor, inputs, label_id)
            _, distance, iters = attack3(model, inputs, target_id, alpha=alpha)
        elif id==3:
            _, distance, iters = attack4(model, inputs, label_id, alpha=alpha)

        # results = pd.concat([results, pd.DataFrame({f'attack {id} distance': [distance], f'attack {id} iters': [iters]})])
        results = pd.concat([results, pd.DataFrame({'distance': [distance], 'iters': [iters]})])
        if (i+1)%100==0:
            print(f"Attack {id} statistics at iteration {i+1}")
            print("success rate:", len(results.dropna())/len(results))
            print(results.describe())
    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--alpha', type=float, default=100)
    args = parser.parse_args()
    
    print("alpha:", args.alpha)
    if args.debug:
        for i in range(100):
            print("example",i)
            debug_example(i, args.alpha)
        exit()
    results_dir = 'results'

    data_dir = 'data/adv_train'
    os.makedirs(f'{results_dir}/tensors', exist_ok = True)
    os.makedirs(f'{results_dir}/images', exist_ok = True)

    results = run_and_compare_attacks(model, mnist, args.id, args.alpha, range(args.n))
    print(f"Attack {args.id} final statistics:")
    print("Success rate:", len(results.dropna())/len(results))
    print(results.describe())
    results.to_csv(f'{results_dir}/compare_attacks-{args.id}.csv')
