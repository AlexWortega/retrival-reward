from datasets import load_dataset
from tqdm.contrib import tzip
from torch.utils.data import Dataset


def load_qs_and_ans():
    dataset = load_dataset('neulab/conala', 'curated')

    qs = dataset['train']['rewritten_intent']
    ans = dataset['train']['snippet']

    return qs, ans


def dataset_for_retrieval():
    qs, ans = load_qs_and_ans()

    data_qa = []
    for question, answer in zip(qs, ans):
        data_qa += [f"Question: {question} Answer: {answer} <|endoftext|>"]
    return data_qa


class QADataset(Dataset):
    def __init__(self, tokenizer):
        qs, ans = load_qs_and_ans()

        self.tokenized = []

        for a, q in tzip(ans, qs):
            pr = f'Question: {q} Answer: {a} {tokenizer.eos_token}'

            enc = self._encode(text=pr, tokenizer=tokenizer)
            self.tokenized += [enc]

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, item):
        return self.tokenized[item]

    @staticmethod
    def _encode(text, tokenizer):
        encoded_sample = tokenizer.encode(text, padding='max_length', max_length=512, return_tensors='pt')

        return encoded_sample
