from typing import List, Dict, Tuple

import pandas as pd
import torch
from sklearn.metrics import classification_report

RESULTS_PATH = "./pos_data/result.csv"


def read_results_csv() -> Tuple:
    frame = pd.read_csv(RESULTS_PATH, header=None)
    ytrue = frame[frame.columns[1]]
    ypred = frame[frame.columns[2]]
    return list(ytrue), list(ypred)


def remove_pads(padded_list: List) -> List:
    return padded_list[1:-1]


def evaluate(model, iterator, idx2tag, tag2idx: Dict) -> None:
    print('Starting eval')
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        print('Generating all predictions from the test set')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    with open(RESULTS_PATH, 'w') as file_out:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words) == len(tags)
            for w, t, p in zip(
                    remove_pads(words),
                    remove_pads(tags),
                    remove_pads(preds)):
                file_out.write("\"{}\",\"{}\",\"{}\"\n".format(w, t, p))
            file_out.write("\n")

    y_true, y_pred = read_results_csv()
    report = classification_report(y_true, y_pred)
    print(report)
