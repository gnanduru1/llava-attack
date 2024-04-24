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

model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True)

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: "
example = mnist[0][0]
label = mnist[0][1] # This is 7
target = '0' # I want my code to misclassify example as a 0

example.save('example.png')

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_id)

img = example

inputs = processor(prompt, img, return_tensors='pt').to(0, torch.float16)
inputs['pixel_values'].requires_grad = True

input_ids = inputs['input_ids']

# outputs = model(**inputs)
# output_logits = outputs.logits

target_string = "0"
target = processor(target_string, return_tensors="pt")

target_id = target['input_ids'][0,-1]

# output_ids = torch.argmax(output_logits, dim=-1)

s = -1
# digit_logits = output_logits[:,s]
# digit = processor.decode(torch.argmax(digit_logits))
# print("decoded digit:", digit) # This returns 7, and I want to perturb the image for it to return 0

loss_fn = torch.nn.CrossEntropyLoss()
# print(digit_logits.shape)
# loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))
# print(loss)

for i in range(num_iterations):
    outputs = model(**inputs)
    output_logits = outputs.logits

    digit_logits = output_logits[:, s]
    loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))

    loss.backward()

    with torch.no_grad():
        inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
        inputs['pixel_values'].grad.zero_()

    digit = processor.decode(torch.argmax(digit_logits))
    print(f"Iteration {i+1}: Decoded digit: {digit}, loss: {loss}")