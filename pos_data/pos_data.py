from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

import nltk


def create_pos_annotations() -> List:
    tagged_sents = nltk.corpus.treebank.tagged_sents()
    print('Length of tagged sentences', len(tagged_sents))
    return tagged_sents


def build_tags_and_ids(tagged_sents: List) -> Tuple[Dict, Dict]:
    tags = list(set(word_pos[1] for sent in tagged_sents for word_pos in sent))

    tags = ["<pad>"] + tags

    tag2idx = {tag: idx for idx, tag in enumerate(tags)}
    idx2tag = {idx: tag for idx, tag in enumerate(tags)}
    return tag2idx, idx2tag


def create_training_splits(tagged_sents: List) -> Tuple:
    train_data, test_data = train_test_split(tagged_sents, test_size=.1)
    print('Training data split length ', len(train_data))
    print('Test data split length ', len(test_data))
    return train_data, test_data
