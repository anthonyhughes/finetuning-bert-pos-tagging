import argparse


def parse_train_arguments():
    parser = argparse.ArgumentParser(description='BERT for POS detection')
    parser.add_argument('--trained_model_path',
                        default='./pos_data/trained-model.torch',
                        type=str,
                        help='the final trained model is stored here')
    parser.add_argument('--training_steps',
                        default=500,
                        type=int,
                        help='select a number of steps to stop the training at')
    parser.add_argument('--evaluation_only',
                        default=False,
                        type=bool,
                        help='run the evaluation only')
    parser.add_argument('--classification_report_only',
                        default=False,
                        type=bool,
                        help='run the classification report only')
    args = parser.parse_args()
    return args
