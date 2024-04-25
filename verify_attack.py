from util import seed_everything
seed_everything()

import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torchvision.io as io
from PIL import Image

# Set the number of iterations and the step size
num_iterations = 100
step_size = 0.01

run_path = 'results/run1'
model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        local_files_only=True
    )
correct = 0
total = 0

for ind in range(100):
    image_path = f"{run_path}/images/{ind}.png"
    tensor_path = f"{run_path}/tensors/{ind}.pt"
    example = Image.open(image_path)
    label = 5
    #example.save('example2.png')

    prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "

    processor = AutoProcessor.from_pretrained(model_id)
    img = example

    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    inputs['pixel_values'] = torch.load(tensor_path)

    input_ids = inputs['input_ids']

    outputs = model(**inputs)
    #outputs = model(**inputs)
    output_logits = outputs.logits

    #s = input_ids.shape[1]+1
    #print(s)

    #output = processor.decode(torch.argmax(output_logits))
    output_ids = torch.argmax(output_logits, dim=-1)

    s = -1
    #need to figure this out
    digit_logits = output_logits[:,s]
    digit = processor.decode(torch.argmax(digit_logits))
    total += 1
    if str(digit) == str(label):
        correct +=1
    print(ind, digit, label, f"{correct}/{total}, {correct/total}")
