from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch 
import argparse
from summary_dataset import Dataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='T5 model training and alignment')
    parser.add_argument('--train_data', type=str, default='automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/train', help='train data folder')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--load_model', type=str, default='t5-small', help='path to T5 model to load')
    parser.add_argument('--save_model', type=str, default='T5_custom', help='path to T5 model to save')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(args.load_model)

    optimizer = torch.optim.Adam([
                        {'params': model.parameters()}
                    ], lr=0.01)

    train_dataset = Dataset(args.train_data, tokenizer, model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print('training...')
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            
            inputs, labels = data

            inputs = inputs.view(-1, 512)
            labels = labels.view(-1, 20)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=labels)

            loss = outputs[0]
            print('train loss', loss)
            loss.backward()
            optimizer.step()
            model.save_pretrained(args.save_model)
            print(args.save_model)
