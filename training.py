import argparse
import os
import glob
from attack_util import get_mnist_instance, seed_everything, get_model_and_processor, get_mnist_dataset, get_mnist_torchvision
import torch
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from torch import Tensor
from datasets import Dataset
import pandas as pd
from torchvision.transforms.functional import to_pil_image

def train(model, data, num_epochs=1, batch_size=32):
    # dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}", end='')
        for i in tqdm(range(0,len(data),batch_size)):
            batch = data[i:i+batch_size]
            label_ids = Tensor(batch['label_id']).view(-1).long()
            input_ids = Tensor(batch['input_ids']).squeeze(1).long()
            attention_mask = Tensor(batch['attention_mask']).squeeze(1).long()
            pixel_values = Tensor(batch['pixel_values']).squeeze(1)
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            digit_logits = outputs.logits[:,-1]
            loss = loss_fn(digit_logits, label_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def get_rad_data(rad_data_dir, processor):
    file_list = glob.glob(os.path.join(rad_data_dir, '*.pt'))
    rad_data = dict(pixel_values=[], input_ids=[], attention_mask=[], label_id=[])
    for file in file_list:
        index, label = file.split('/')[-1].split('-')
        image_tensor = torch.load(file).squeeze()
        image_pil = to_pil_image(image_tensor)
        prompt, label_id = get_mnist_instance((image_pil, label), processor)
        for key in prompt.keys():
            rad_data[key].append(prompt[key])
        rad_data['label_id'].append(label_id)
    return Dataset.from_dict(rad_data)

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        label_ids = Tensor(data['label_id']).view(-1).long()
        input_ids = Tensor(data['input_ids']).squeeze(1).long()
        attention_mask = Tensor(data['attention_mask']).squeeze(1).long()
        pixel_values = Tensor(data['pixel_values']).squeeze(1)

        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        digit_logits = outputs.logits[:,-1]
        loss = torch.nn.CrossEntropyLoss()(digit_logits, label_ids)

        digits = torch.argmax(digit_logits, 1)
        correct = torch.sum(digits==label_ids)
        accuracy = correct/label_ids.shape[-1]
        return loss, accuracy


if __name__ == '__main__':
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('-n','--n', type=int, default=1534)
    #1534 is the minimum number of successful attacks for each of the attacks
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--rad', action='store_true', default=False)
    args = parser.parse_args()

    model, processor = get_model_and_processor()
    batch_size = 32
    rad_data_dir = f"rad_data/tensors-{args.id}"
    eval_size = args.n
    mnist_train = get_mnist_dataset(processor, split=f'train[:{args.n}]')
    mnist_test = get_mnist_dataset(processor, split=f'test[:{eval_size}]')
    
    print("Preliminary evaluation")
    loss_before, accuracy_before = evaluate(model, mnist_test)
    print("\tCross-entropy loss:", loss_before.item())
    print("\tAccuracy:", accuracy_before.item())
    print(f"{'RAD ' if args.rad else ''}Training:")
    if args.rad:
        # should we handle failed attacks
        train(model, mnist_train, num_epochs=1, batch_size=batch_size)
        rad_data = get_rad_data(rad_data_dir, processor)[:args.n]
        train(model, rad_data, num_epochs=1, batch_size=batch_size)
    else:
        train(model, mnist_train, num_epochs=2, batch_size=batch_size)

    print("Post-training evaluation")
    loss_after, accuracy_after = evaluate(model, mnist_test)
    print("\tCross-entropy loss:", loss_after.item())
    print("\tAccuracy:", accuracy_after.item())

    data = {
        'loss before': [loss_before.item()],
        'accuracy before': [accuracy_before.item()],
        'loss after': [loss_after.item()],
        'accuracy after': [accuracy_after.item()],
        'rad': [args.rad],
        'attack id': [args.id]
    }
    df = pd.DataFrame(data)

    results_dir = 'results'
    os.makedirs(f'{results_dir}', exist_ok = True)
    if args.rad:
        out = f'results/rad-train-{args.id}'
    else:
        out = f'results/reg-train-{args.id}'
    df.to_csv(out)