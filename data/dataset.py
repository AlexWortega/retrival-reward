from datasets import load_dataset
from tqdm.contrib import tzip
from torch.utils.data import Dataset

def loading(conf):
    return load_dataset(conf.args.dataset, conf.args.parameter_for_dataset)


def load_qs_and_ans(conf):
    dataset = loading(conf)

    qs = dataset['train'][conf.args.question_col]
    ans = dataset['train'][conf.args.answer_col]

    return qs, ans

def load_qs(conf):
    dataset = loading(conf)

    qs = dataset['train'][conf.args.question_col]

    return qs

def load_train(conf):
    dataset = loading(conf)

    train_dataset = dataset['train']

    return train_dataset

def dataset_for_retrieval(conf):
    qs, ans = load_qs_and_ans(conf)

    data_qa = []
    for question, answer in zip(qs, ans):
        data_qa += [f"Question: {question} Answer: {answer} <|endoftext|>"]
    return data_qa


class QADataset(Dataset):
    def __init__(self, tokenizer, conf):
        qs, ans = load_qs_and_ans(conf)

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

class QDataset(Dataset):
    def __init__(self, tokenizer, conf):
      
      qs = load_qs(conf)
     

      self.tokenized = []
      
      for q in qs:

        pr = f'Question: {q} Answer:'

        enc = self._encode(text=pr, tokenizer=tokenizer)
        self.tokenized +=[enc]

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, item):
        return (self.tokenized[item])

    def _encode(self, text, tokenizer):
        
        encoded_sample = tokenizer.encode(text,  padding='max_length', max_length=128, return_tensors='pt')

        return encoded_sample
    
class SumDataset(Dataset):
    def __init__(self, tokenizer, conf):
      qs, ans = load_qs_and_ans(conf)

      self.tokenized = []
      self.answer = []

      for a, q in tzip(ans,qs):

        pr = f'Question: {q}  Answer:'

        enc = self._encode(text=pr, tokenizer=tokenizer)#@, self._encode(text=q, tokenizer=tokenizer), self._encode(text=a, tokenizer=tokenizer)
        self.tokenized += [enc]
        self.answer += [a]

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, item):
        return ([self.tokenized[item], self.answer[item]])

    def _encode(self, text, tokenizer):
        
        encoded_sample = tokenizer.encode(text, padding='max_length', max_length=512, return_tensors='pt')

        return encoded_sample