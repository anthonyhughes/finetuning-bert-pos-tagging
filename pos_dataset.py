from torch.utils import data


class PosDataset(data.Dataset):
    """
    Part of speech torch-based dataset
    """
    SPECIAL_CHARACTERS = ("[CLS]", "[SEP]")

    def __init__(self, labeled_sentences, tokenizer, tag_to_idx):
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        sentences, tags_li = [], []  # list of lists

        for sent in labeled_sentences:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sentences.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])

        self.labeled_sentences, self.tags_li = sentences, tags_li
        print('Dataset init complete')

    def __len__(self):
        return len(self.labeled_sentences)

    def __getitem__(self, idx):
        words = self.labeled_sentences[idx]
        tags = self.tags_li[idx]

        x = []
        y = []

        is_heads = []

        for word, token in zip(words, tags):
            tokens = self.tokenizer.tokenize(word) if word not in self.SPECIAL_CHARACTERS else [word]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            token = [token] + ["<pad>"] * (len(tokens) - 1)  # applied padding to sequence length

            yy = [self.tag_to_idx[each] for each in token]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x) == len(y) == len(is_heads)
        print(f"length of x={len(x)}, length of y={len(y)}, heads{len(is_heads)}")

        sequence_length = len(y)
        return words, x, is_heads, tags, y, sequence_length
