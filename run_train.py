import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from torch import nn
from torch.utils import data
from transformers import BertTokenizer

from model import SequenceClassificationModel
from pos_data.pos_data import build_tags_and_ids, create_training_splits, create_pos_annotations
from pos_dataset import PosDataset
from user_args import parse_train_arguments
from utils.evaluate import evaluate, read_results_csv
from utils.model_utils import pad
from utils.train import train

PATH = './pos_data/trained-model.torch'


def run_train(training_steps: int, trained_model_path: str,
              evaluation_only: bool, classification_report_only: bool) -> None:
    """
    Main function for training and evaluating the model
    @:param training_steps: steps
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating datasets w/ labels')
    tags = create_pos_annotations()
    tag_to_idx, idx2tag = build_tags_and_ids(tags)
    train_data, test_data = create_training_splits(tags)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
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

    if evaluation_only is False:
        model = SequenceClassificationModel(vocab_size=len(tag_to_idx), device=device)
        model.to(device)
        model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        train(model, train_iter, optimizer, criterion)
        torch.save(model.state_dict(), PATH)
        evaluate(model, test_iter, idx2tag, tag_to_idx)
    elif classification_report_only is True:
        y_true, y_pred = read_results_csv()
        report = classification_report(y_true, y_pred)
        print(report)
    else:
        model = SequenceClassificationModel(vocab_size=len(tag_to_idx), device=device)
        model.load_state_dict(torch.load(PATH), strict=False)
        evaluate(model, test_iter, idx2tag, tag_to_idx)


if __name__ == '__main__':
    args = parse_train_arguments()
    run_train(args.training_steps,
              args.trained_model_path,
              args.evaluation_only,
              args.classification_report_only)
