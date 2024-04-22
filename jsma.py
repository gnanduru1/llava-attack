import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageFilter




model_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=True, download=True,
                      )

example = mnist[0][0]
example.save('example.png')

prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT:"



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

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
