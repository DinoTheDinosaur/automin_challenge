from torch.utils.data import IterableDataset, DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import torch

class Dataset(Dataset):
    def __init__(self, train_folder, tokenizer, model, train_size=512, label_size=40, summary_num=10):
        self.train_size = train_size
        self.label_size = label_size
        self.summary_num = summary_num
        self.tokenizer = tokenizer
        self.model = model 
        self.data, self.labels = self.collect_all_data(train_folder)
        # TODO: parse labels to get summaries (for better alignment)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def collect_all_data(self, train_folder):
        all_chunks = []
        all_labels = []
        for folder in os.listdir(train_folder):
            print("folder", folder)
            for txt_file in os.listdir(os.path.join(train_folder, folder)):
                if 'transcript' in txt_file:
                    train_file = os.path.join(train_folder, folder, txt_file)
                elif 'GENER' in txt_file:
                    summary_file = os.path.join(train_folder, folder, txt_file)

            train_chunks = self.read_file(train_file, self.train_size)
            label_chunks = self.read_file(summary_file, self.label_size)

            train_chunks, train_labels = self.align_chunks(train_chunks, label_chunks, self.summary_num)

            all_chunks.append(train_chunks)
            all_labels.append(train_labels)

            #if len(all_chunks) == 32:
            #    break 

        return torch.cat(all_chunks, dim=0), torch.cat(all_labels, dim=0)

    def read_file(self, txt_file, size):
        with open(txt_file, 'r+') as f:
            transcript = f.read().replace('\n', ' ')
            input_ids = self.tokenizer(transcript, return_tensors='pt').input_ids
            chunks = self.separate_into_chunks(input_ids, size)
            return chunks

    @staticmethod
    def separate_into_chunks(input_ids, size):
        chunks = []
        for i in range(0, input_ids.shape[1], size):
            chunk_ids = input_ids[:, i : i + size]
            if chunk_ids.shape[1] == size:
                chunks.append(chunk_ids)
        return chunks

    def align_chunks(self, train_chunks, label_chunks, summary_num):
        data = []
        losses = []
        for i in range(0, len(train_chunks)):
            for j in range(0, len(label_chunks)):
                loss = self.model(input_ids=train_chunks[i], labels=label_chunks[j])[0].item()
                data.append((train_chunks[i], label_chunks[j]))
                losses.append(loss)

        min_losses = sorted(range(len(losses)), key = lambda sub: losses[sub])[:summary_num]
        all_train_chunks = []
        all_train_labels = []
        for i in min_losses:
            curr_train_chunks, curr_label_chunks = data[i]

            all_train_chunks.append(curr_train_chunks)
            all_train_labels.append(curr_label_chunks)

        return torch.cat(all_train_chunks), torch.cat(all_train_labels)

if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    train_dataset = Dataset('automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/train')
    train_loader = DataLoader(train_dataset, tokenizer, model, batch_size=4, shuffle=True)

