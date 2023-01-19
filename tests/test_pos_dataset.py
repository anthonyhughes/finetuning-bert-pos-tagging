import nltk

from utils.model_utils import pad
from pos_dataset import PosDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils import data

tagged_sentences = nltk.corpus.treebank.tagged_sents()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels = list(set(label
                  for sentence in tagged_sentences
                  for token, label in sentence)
              )

labels = [""] + labels

tag_to_idx = {tag: idx for idx, tag in enumerate(labels)}
idx_to_tag = {idx: tag for idx, tag in enumerate(labels)}

train_data, test_data = train_test_split(tagged_sentences, test_size=.1)
train_dataset = PosDataset(train_data, bert_tokenizer, tag_to_idx)
# result = train_dataset.__getitem__(1)
# print(result[0], result[1], result[2], result[3], result[4], sep='\n')

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=pad)

for i, batch in enumerate(train_iter):
    print(batch[0], batch[1], batch[2], batch[3], batch[4], sep='\n')
