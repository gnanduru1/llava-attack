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
from attack_util import get_model_and_processor, save_img, get_mnist_instance, mnist, llava_id

model, processor = get_model_and_processor(llava_id)

def get_target(inputs, label_id):
    output_logits = model(**inputs).logits
    digit_ids = processor([str(i) for i in range(10)])['input_ids'][:,-1]
    max_likelihood = 0
    target_id = 0
    for digit_id in digit_ids:
        if digit_id == label_id:
            continue
        likelihood = output_logits[0,-1,digit_id.item()]
        if likelihood >= max_likelihood:
            max_likelihood = likelihood
            target_id = digit_id
    return target_id


def loss_fn_1(logits, target, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, target)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    return adv_loss + alpha*regularizer


def attack1(model, inputs, target, num_iterations=100, step_size=0.01, alpha=0.1, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss_fn_1(digit_logits, target.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id == target:
            return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), i+1
    return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), num_iterations


def loss_fn_2(logits, label, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, label)
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    return -adv_loss + alpha*regularizer


def attack2(model, inputs, label, num_iterations=100, step_size=0.01, alpha=0.1, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss_fn_2(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id != label:
            return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), i+1
    return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), num_iterations

def debug_example(i):
    inputs, label_id = get_mnist_instance(mnist[i], processor)
    save_img(inputs['pixel_values'], inputs['pixel_values'], f'image_{i}', results_dir)

    target_id = get_target(inputs, label_id)
    print(f"Attack 1, target = {processor.decode(target_id)}")
    new_img, distance_1, iters_1 = attack1(model, inputs, target_id, debug=True)

    inputs, label_id = get_mnist_instance(mnist[i], processor)
    print("Attack 2")
    new_img, distance_2, iters_2 = attack2(model, inputs, label_id, debug=True)

    print(f"Attack 1 took {iters_1} iterations")
    print(f"Attack 1 distance: {distance_1}")
    print(f"Attack 2 took {iters_2} iterations")
    print(f"Attack 2 distance: {distance_2}")
    save_img(new_img, new_img, f'perturbed_image_{i}', results_dir)


def run_and_compare_attacks(model, mnist, id, data_range=range(3000)):
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = torch.nn.DataParallel(model)
    results = pd.DataFrame(columns=[f'attack {id} distance', f'attack {id} iters'])
    
    if id==0:
        print("Attack 1")
    elif id==1:
        print("Attack 2")
    else:
        print(f"Invalid id: {id}")
        return
    
    for i in tqdm(data_range):
        row = mnist[i]
        inputs, label_id = get_mnist_instance(row, processor)
        
        if id==0:
            target_id = get_target(inputs, label_id)
            _, distance, iters = attack1(model, inputs, target_id)
        elif id==1:
            _, distance, iters = attack2(model, inputs, label_id)

        results = pd.concat([results, pd.DataFrame({'attack {id} distance': [distance], 'attack {id} iters': [iters]})])
    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n', type=int, default=3000)
    args = parser.parse_args()
    
    results_dir = 'results'

    data_dir = 'data/adv_train'
    os.makedirs(f'{results_dir}/tensors', exist_ok = True)
    os.makedirs(f'{results_dir}/images', exist_ok = True)

    # debug_example(0)

    results = run_and_compare_attacks(model, mnist, args.id, range(args.n))
    print(results.describe())
    results.to_csv(f'{results_dir}/compare_attacks-{args.id}.csv')
