import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageFilter

# Set the number of iterations and the step size
num_iterations = 100
step_size = 0.01

evaluate_file = 'logs/evaluate_test.txt'
model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=False, download=True)
model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        local_files_only=True
    )
correct = 0
total = 0

for number in range(len(mnist)):
    example = mnist[number][0]
    label = mnist[number][1]
    #example.save('example2.png')

    prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "

    processor = AutoProcessor.from_pretrained(model_id)
    img = example

    inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
    inputs['pixel_values'].requires_grad = True

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
    #print("decoded digit:", digit)
    #print(digit, label)
    total += 1
    if str(digit) == str(label):
        correct +=1
    #print(f"{correct}/{total}, {correct/total}")

    with open(evaluate_file, 'a') as f:
        f.write(f"{digit} {label}\n")

with open(evaluate_file, 'a') as f:
    f.write(f"{correct}, {total}, {correct/total}")