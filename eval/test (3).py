import argparse
import os
import torch
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead
from bert_score import score 
from rouge_score import rouge_scorer


class SumDataset(Dataset):
    def __init__(self, tokenizer, dataset_name):
        dataset = load_dataset("gsm8k",'main')
        qs = dataset['test']['question']
        ans = dataset['test']['answer']

        self.tokenized = []
        self.answer = []

        for a, q in tqdm(zip(ans, qs), total=len(ans)):
            pr = f'Question: {q} ? Answer:'
            enc = tokenizer.encode(pr, a, padding='max_length', max_length=512, return_tensors='pt')
            self.tokenized.append(enc)
            self.answer.append(a)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, item):
        return self.tokenized[item], self.answer[item]


def evaluate(model, tokenizer, test_dataloader, device, gen_kwargs):
    scores_bert = []
    scores_rouge1 = []
    scores_rouge2 = []
    ans = []
    ans_gt = []
    for batch in tqdm(test_dataloader):
        t = batch[0].to(device)
        gen = model.generate(t.squeeze(1), **gen_kwargs)
        a = tokenizer.batch_decode(gen)
        batch_gen = [i.replace('', '').split('Answer:')[1].lower() for i in a]
        batch_orig = [i.lower() for i in batch[1]]
        ans += batch_gen
        ans_gt += batch_orig

        # Calculate BERT score
        P, R, F1 = score(batch_gen, batch_orig, lang='en', verbose=False)
        scores_bert.append(np.mean(F1.cpu().numpy()))

        # Decode the tokenized strings back to regular strings
        batch_gen_decoded = [tokenizer.decode(tokenizer.encode(text)) for text in batch_gen]
        batch_orig_decoded = [tokenizer.decode(tokenizer.encode(text)) for text in batch_orig]

        # Calculate ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        for b_g, b_o in zip(batch_gen_decoded, batch_orig_decoded):
            scores = scorer.score(b_g, b_o)
            scores_rouge1.append(np.mean([s['rouge1'].fmeasure for s in scores]))
            scores_rouge2.append(np.mean([s['rouge2'].fmeasure for s in scores]))
        """

    bleu_score = nltk.translate.bleu_score.corpus_bleu(
        list_of_references=[ans_gt], hypotheses=[ans]
    ) * 100
    avg_bert_score = np.mean(scores_bert) * 100
    #rouge1_score = np.mean(scores_rouge1) * 100
    #rouge2_score = np.mean(scores_rouge2) * 100

    metrics = pd.DataFrame({
        'Dataset': [args.dataset_name],
        'Model Path': [args.model_path],
        'BLEU': [bleu_score],
        'BERT Score': [avg_bert_score],
        #'ROUGE-1': [rouge1_score],
        #'ROUGE-2': [rouge2_score]
    })

    metrics_file = f"Metrics_{os.path.basename(args.model_path)}_{args.dataset_name}.csv"
    if os.path.exists(metrics_file):
        metrics.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a TRL model on a summarization dataset')
    parser.add_argument('model_path', type=str, help='path to the TRL model checkpoint')
    parser.add_argument('dataset_name', type=str, help='name of the dataset to use for evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    model.to(device)

    gen_kwargs = {
        "min_length": 20,
        "max_length": 768,
        "top_k": 10,
        "top_p": 1,
        "do_sample": True,
        "no_repeat_ngram_size": 2,
        "pad_token_id": tokenizer.eos_token_id
    }

    data = SumDataset(tokenizer, args.dataset_name)
    test_dataloader = DataLoader(data, batch_size=10)

    evaluate(model, tokenizer, test_dataloader, device, gen_kwargs)
