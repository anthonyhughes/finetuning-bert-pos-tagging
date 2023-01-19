from typing import List

import numpy as np
import torch


def remove_pads(padded_list: List) -> List:
    return padded_list[1:-1]


def evaluate(model, iterator, idx2tag, tag2idx) -> None:
    print('Starting eval')
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    with open("result", 'w') as file_out:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(
                    remove_pads(words.split()),
                    remove_pads(tags.split()),
                    remove_pads(preds)):
                file_out.write("{} {} {}\n".format(w, t, p))
            file_out.write("\n")

    # calc metric
    y_true = np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print(f"Total accuracy=${acc}")
