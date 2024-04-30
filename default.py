import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets, transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image, ImageFilter
import argparse

MODEL_SHORTHAND = {
    'llava': "llava-hf/llava-1.5-7b-hf",
    'blip': "Salesforce/blip2-opt-2.7b",
}

PROMPTS = {
    "llava-hf/llava-1.5-7b-hf": "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: ",
    "Salesforce/blip2-opt-2.7b": "this is a picture of the number",
           }

def main(model_id, device):
    #mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=False, download=True)
    #example = mnist[0][0]

    num = 35
    image_path = f"results/run1/images/{num}.png"
    tensor_path = f"results/run1/tensors/{num}.pt"

    img = Image.open(image_path)

    prompt = PROMPTS[model_id]

    model, inputs = None, None
    processor = AutoProcessor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        )

    if model_id == "llava-hf/llava-1.5-7b-hf":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            quantization_config=bnb_config
        )
        inputs = processor(prompt, img, return_tensors='pt').to(device, torch.float16)
        inputs['pixel_values'] = torch.load(tensor_path)

    elif model_id == "Salesforce/blip2-opt-2.7b":
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            quantization_config=bnb_config
        )

        transform = transforms.Compose([transforms.Resize(224)])
        inputs = processor(img,prompt, return_tensors='pt').to(device, torch.float16)
        inputs['pixel_values'] = transform(torch.load(tensor_path))

    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"[{model_id}]:", processor.decode(output[0], skip_special_tokens=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str.lower, default="llava-hf/llava-1.5-7b-hf", help="Model to run. Supported models: llava, blip"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA Device"
    )

    args = parser.parse_args()
    if args.model in MODEL_SHORTHAND:
        args.model = MODEL_SHORTHAND[args.model]
    if args.model not in PROMPTS:
        raise ValueError('Supported models: llava, blip')
    
    print("Command-line args:", args)
    main(model_id=args.model, device=args.device)
