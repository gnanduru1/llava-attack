import os
from getpass import getuser
import torch
from torchvision import datasets, utils as vutils
from transformers import AutoProcessor, LlavaForConditionalGeneration
from attack_util import seed_everything

# Environment initialization
seed_everything()
user_name = getuser()
datasets_path = f'/scratch/{user_name}/datasets'
os.environ['HF_HOME'] = datasets_path

# Load CIFAR10 dataset
cifar_dataset = datasets.CIFAR10(datasets_path, train=False, download=True)

# Load model and processor
model_identifier = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_identifier, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    load_in_4bit=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_identifier)

# Attack configuration
num_iterations = 500
step_size = 0.1
results_directory = 'results/cifar'
os.makedirs(f'{results_directory}/tensors', exist_ok=True)
os.makedirs(f'{results_directory}/images', exist_ok=True)

# Tokenization of class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
tokenized_labels = {label: processor.tokenizer(label, add_special_tokens=False)['input_ids'] for label in labels}

# Initialize target and input data
image, label_index = cifar_dataset[0]
actual_label = labels[label_index]

# Helper functions
def moving_average(x, tensor):
    result = []
    for i in range(x):
        # Compute mean from index i to tensor.shape[0]-x+i
        mean_val = tensor[i:tensor.shape[0]-x+i].mean(dim=0)
        result.append(mean_val)
    return torch.stack(result)


def class_probabilities_from_logits(logits):
    probabilities = {}
    for label, token_ids in tokenized_labels.items():
        sequence_length = logits.shape[1]
        if sequence_length >= len(token_ids):
            relevant_logits = moving_average(len(token_ids), logits[0])
            label_logits = torch.tensor([relevant_logits[idx, token_id] for idx, token_id in enumerate(token_ids)])
            probabilities[label] = label_logits.mean()
    return probabilities

# Preparing inputs for the model
prompt_text = f"USER: <image>\nFrom the following list of options: {', '.join(labels)}, what is this image showing?\nASSISTANT: This is a "
inputs = processor(prompt_text, image, return_tensors='pt').to(0, torch.float16)
inputs['pixel_values'].requires_grad = True

generated_output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(generated_output[0][2:], skip_special_tokens=True))

outputs = model(**inputs)
logits = outputs.logits
probabilities = class_probabilities_from_logits(logits)
print("first pred", max(probabilities, key=probabilities.get))
del probabilities[actual_label]

target_label = min(probabilities, key=probabilities.get)
target_token_ids = processor.tokenizer(target_label, add_special_tokens=False)['input_ids']
target_input_ids = processor(target_label, return_tensors="pt")['input_ids'][0, -len(target_token_ids):]


# Adversarial attack iterations
for iteration in range(num_iterations):
    outputs = model(**inputs)
    logits = outputs.logits

    mean = moving_average(len(target_token_ids), logits[0])
    loss = torch.nn.CrossEntropyLoss()(mean, target_input_ids.view(-1).to(mean.device))
    loss.backward()

    with torch.no_grad():
        inputs['pixel_values'] -= step_size * inputs['pixel_values'].grad
        inputs['pixel_values'].grad.zero_()

    # Logging and saving results
    vutils.save_image(inputs['pixel_values'], f'{results_directory}/images/{iteration}.png')
    torch.save(inputs['pixel_values'], f'{results_directory}/tensors/{iteration}.pt')
    probabilities = class_probabilities_from_logits(logits)
    predicted_label = max(probabilities, key=probabilities.get)
    print(f"Iteration {iteration + 1}: loss: {loss}, Predicted: {predicted_label}, Actual: {actual_label}, Target: {target_label}")

    if predicted_label == target_label:
        print("Target prediction verified.")
        generated_output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(processor.decode(generated_output[0][2:], skip_special_tokens=True))
        break
