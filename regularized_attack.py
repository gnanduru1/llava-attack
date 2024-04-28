# from util import seed_everything
# seed_everything()

import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import pandas as pd

model_id = "llava-hf/llava-1.5-7b-hf"

def get_model_and_processor(model_id):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        local_files_only=True
    )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

model, processor = get_model_and_processor(model_id)

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
    return target_id, processor.decode(target_id)


def loss_fn_1(logits, target, original_image, current_image, alpha=0.1):
    adv_loss = torch.nn.CrossEntropyLoss()(logits, target)
    # regularizer = torch.nn.L1Loss()(original_image, inputs['pixel_values'])
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
    # regularizer = torch.nn.L1Loss()(original_image, inputs['pixel_values'])
    regularizer = torch.nn.MSELoss()(original_image, current_image)
    return adv_loss - alpha*regularizer


def attack2(model, inputs, label, num_iterations=100, step_size=0.01, alpha=0.1, debug=False):
    inputs['pixel_values'].requires_grad = True
    original_image = inputs['pixel_values'].detach().clone()
    for i in range(num_iterations):
        output_logits = model(**inputs).logits

        digit_logits = output_logits[:, -1]
        loss = loss_fn_2(digit_logits, label.view(-1).to(digit_logits.device), original_image, inputs['pixel_values'], alpha=alpha)

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] += step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit_id = torch.argmax(digit_logits)
        if debug:
            print(f"Iteration {i+1}: Decoded digit: {processor.decode(digit_id)}, loss: {loss}")
        if digit_id != label:
            return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), i+1
    return inputs['pixel_values'], torch.nn.MSELoss()(original_image, inputs['pixel_values']).item(), num_iterations

def get_data(row):
    img, label = row
    #img.save('example.png')
    prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "
    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    label_id = processor(str(label))['input_ids'][0, -1]
    return inputs, label_id

def debug_example(i):
    inputs, label_id = get_data(mnist[i])
    vutils.save_image(inputs['pixel_values'], f'{results_dir}/images/image_{i}.png')
    torch.save(inputs['pixel_values'], f'{results_dir}/tensors/image_{i}.pt')

    print("Attack 2")
    new_img, distance_2, iters_2 = attack2(model, inputs, label_id, debug=True)

    inputs, label_id = get_data(mnist[i])

    target_id, target_string = get_target(inputs, label_id)
    print(f"Attack 1, target = {target_string}")
    new_img, distance_1, iters_1 = attack1(model, inputs, target_id, debug=True)


    print(f"Attack 1 took {iters_1} iterations")
    print(f"Attack 1 distance: {distance_1}")
    print(f"Attack 2 took {iters_2} iterations")
    print(f"Attack 2 distance: {distance_2}")

    vutils.save_image(new_img, f'{results_dir}/images/perturbed_image_{i}.png')
    torch.save(new_img, f'{results_dir}/tensors/perturbed_image_{i}.pt')



def run_and_compare_attacks(model, mnist):
    results = pd.DataFrame(columns=['attack 1 distance', 'attack 1 iters', 'attack 2 distance', 'attack 2 iters'])
    for row in mnist:
        inputs, label_id = get_data(row)
        target_id, target_string = get_target(inputs, label_id)
        _, distance_1, iters_1 = attack1(model, inputs, target_id)
        inputs, label_id = get_data(row)
        _, distance_2, iters_2 = attack2(model, inputs, label_id)
        results = pd.concat([results, pd.DataFrame({'attack 1 distance': [distance_1], 'attack 1 iters': [iters_1], 'attack 2 distance': [distance_2], 'attack 2 iters': [iters_2]})])
    return results

if __name__ == '__main__':
    results_dir = 'results/run1'

    if not os.path.exists(f'{results_dir}/tensors'):
        os.makedirs(f'{results_dir}/tensors', exist_ok = True)
    if not os.path.exists(f'{results_dir}/images'):
        os.makedirs(f'{results_dir}/images', exist_ok = True)
    mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)

    debug_example(0)

    results = run_and_compare_attacks(model, mnist)
    print(results.summary())
    results.to_csv(f'{results_dir}/results.csv')
