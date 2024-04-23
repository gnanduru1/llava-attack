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

example = mnist[0][0]
example.save('example.png')

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit "
target = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: This is the digit 0."

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
target_inputs = processor(target, img, return_tensors='pt').to(0, torch.float16)

input_ids = inputs['input_ids']
for i,id in enumerate(input_ids[0]):
    print(i, processor.decode(id))


outputs = model(**inputs)
output_logits = outputs.logits
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))

target_string = "0"
target = processor(target_string, return_tensors="pt")

target_id = target['input_ids'][0,-1]

#s = input_ids.shape[1]+1
#print(s)

#output = processor.decode(torch.argmax(output_logits))
output_ids = torch.argmax(output_logits, dim=-1)
print(output_ids[0].shape)
for i,id in enumerate(output_ids[0]):
    print(i, processor.decode(id))

s = -1
#need to figure this out
digit_logits = output_logits[:,s]
digit = processor.decode(torch.argmax(digit_logits))
print("decoded digit:", digit)

loss_fn = torch.nn.CrossEntropyLoss()
#loss = loss_fn(output_logits.view(-1, output_logits.size(-1)), target_ids.view(-1))
print(digit_logits.shape)
loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))
print(loss)

#loss = outputs.loss
#torch.autograd.grad(loss, inputs['pixel_values'])[0]

#print(f"Loss: {loss.item()}")

