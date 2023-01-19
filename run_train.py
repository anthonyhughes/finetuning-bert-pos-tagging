import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from transformers import BertTokenizer

from model import SequenceClassificationModel
from pos_data.pos_data import build_tags_and_ids, create_training_splits, create_pos_annotations
from pos_dataset import PosDataset
from utils.evaluate import evaluate
from utils.model_utils import pad
from utils.train import train

PATH = './pos_data/trained-model.torch'


def run_train() -> None:
    """
    Main function for running the training of the BERT model
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tags = create_pos_annotations()
    tag_to_idx, idx2tag = build_tags_and_ids(tags)
    train_data, test_data = create_training_splits(tags)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    model = SequenceClassificationModel(vocab_size=len(tag_to_idx), device=device)
    model.to(device)
    model = nn.DataParallel(model)

    train_dataset = PosDataset(train_data, tokenizer, tag_to_idx)
    eval_dataset = PosDataset(test_data, tokenizer, tag_to_idx)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    test_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train(model, train_iter, optimizer, criterion)

    torch.save(model.state_dict(), PATH)

    evaluate(model, test_iter, idx2tag, tag_to_idx)


if __name__ == '__main__':
    run_train()
