import os
from getpass import getuser
os.environ['HF_HOME'] = f'/scratch/{getuser()}/datasets'

import torch
from torchvision import datasets, transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image, ImageFilter
import torchvision.utils as vutils
import argparse

### START ATTACK HERE ###
    
# Set the number of iterations and the step size
num_iterations = 1000
step_size = 0.01
device=0

results_dir = 'results/long_transfer'

if not os.path.exists(f'{results_dir}/tensors'):
    os.makedirs(f'{results_dir}/tensors', exist_ok = True)
if not os.path.exists(f'{results_dir}/images'):
    os.makedirs(f'{results_dir}/images', exist_ok = True)


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        )

llava_id = "llava-hf/llava-1.5-7b-hf"
mnist = datasets.MNIST(f'/scratch/{getuser()}/datasets/mnist', train=False, download=True)

llava_prompt = "USER: <image>\nWhat digit [0-9] is this?\nASSISTANT: "

llava = LlavaForConditionalGeneration.from_pretrained(
        llava_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        quantization_config=bnb_config
    )

llava_processor = AutoProcessor.from_pretrained(llava_id)

blip_id = "Salesforce/blip2-opt-2.7b"
blip = Blip2ForConditionalGeneration.from_pretrained(
    blip_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)
blip_processor = AutoProcessor.from_pretrained(blip_id)
blip_prompt = "this is a picture of the number"
transform = transforms.Compose([transforms.Resize(224)])

for idx in range(1):
    example = mnist[idx][0]
    label = mnist[idx][1]

    print(f"Index: {idx}, label: {label}")

    example.save('example.png')

    img = example

    inputs = llava_processor(llava_prompt, img, return_tensors='pt').to(device, torch.float16)
    inputs['pixel_values'].requires_grad = True

    input_ids = inputs['input_ids']

    # outputs = model(**inputs)
    # output_logits = outputs.logits

    target_string = "0"
    target = llava_processor(target_string, return_tensors="pt")

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
        outputs = llava(**inputs)
        output_logits = outputs.logits

        digit_logits = output_logits[:, s]
        loss = loss_fn(digit_logits, target_id.view(-1).to(digit_logits.device))

        loss.backward()

        with torch.no_grad():
            inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
            inputs['pixel_values'].grad.zero_()

        digit = llava_processor.decode(torch.argmax(digit_logits))
        print(f"LLaVa Iteration {i+1}: Decoded digit: {digit}, loss: {loss}")
        
        blip_inputs = blip_processor(img, blip_prompt, return_tensors='pt').to(device, torch.float16)
        blip_inputs['pixel_values'] = transform(inputs['pixel_values'])
        blip_output = blip.generate(**blip_inputs, do_sample=False)
        print("BLIP response:", blip_processor.decode(blip_output[0], skip_special_tokens=True))


        # if digit == target_string:
        #     print("Verifying")
        #     output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        #     print(processor.decode(output[0][2:], skip_special_tokens=True))

        #     outputs = model(**inputs)
        #     output_logits = outputs.logits

        #     digit_logits = output_logits[:, s]
        #     print(processor.decode(torch.argmax(digit_logits)))

        #perturbed_image = ToPILImage()(inputs['pixel_values'][0])
        #perturbed_image.save(f'results/iteration_{i}.png')
        vutils.save_image(inputs['pixel_values'], f'{results_dir}/images/{i}.png')
        torch.save(inputs['pixel_values'], f'{results_dir}/tensors/{i}.pt')
