import torch


class Const:
    UNK_ID, UNK_TOKEN = 1, "<unk>"
    PAD_ID, PAD_TOKEN = 0, "<pad>"
    PAD_TAG_ID, PAD_TAG_TOKEN = 0, "<pad>"


class Vectorizer(object):
    base_word_to_ix = {
        Const.UNK_TOKEN: Const.UNK_ID,
        Const.PAD_TOKEN: Const.PAD_ID
    }

    base_ix_to_word = {
        Const.UNK_ID: Const.UNK_TOKEN,
        Const.PAD_ID: Const.PAD_TOKEN
    }

    tag_to_ix = {
        Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID
    }

    def __init__(self, texts, tags, word_embedder=None):

        self.max_token_len = max(map(len, tags))
        self.max_char_len = max(map(len, [' '.join(i) for i in texts]))

        print(self.max_token_len)
        print(self.max_char_len)

        tokens = set([token for seq in texts for token in seq])
        self.word2Index = {word: index for index, word in enumerate(sorted(tokens), start=2)}
        self.index2Word = {index: word for index, word in enumerate(sorted(tokens), start=2)}
        self.word2Index = {**self.word2Index, **Vectorizer.base_word_to_ix}
        self.index2Word = {**self.index2Word, **Vectorizer.base_ix_to_word}

        char_tokens = set(list('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'))
        self.char2Index = {str(char): index for index, char in enumerate(sorted(char_tokens), start=2)}
        self.index2Char = {index: str(char) for index, char in enumerate(sorted(char_tokens), start=2)}

        self.char2Index = {**self.char2Index, **Vectorizer.base_word_to_ix}
        self.index2Char = {**self.index2Char, **Vectorizer.base_ix_to_word}
        len_tags = len(Vectorizer.tag_to_ix)
        tags = set([tag for seq in tags for tag in seq])

        self.tags2Index = {tag: len_tags + index for index, tag in enumerate(sorted(tags))}
        self.tags2Index = {**self.tags2Index, **Vectorizer.tag_to_ix}
        self.index2tags = {index: tag for tag, index in self.tags2Index.items()}

        if word_embedder:
            self.embedding_matrix = list()
            for ix in sorted(self.index2Word.keys()):
                self.embedding_matrix.append(word_embedder[self.index2Word[ix]])

    def lookup_index(self, token):
        if token in Vectorizer.base_word_to_ix:
            return Vectorizer.base_word_to_ix[token]
        else:
            return self.word2Index.get(token, Const.UNK_ID)

    def lookup_tag(self, tag):
        return self.tags2Index.get(tag, Const.UNK_ID)

    def size(self):
        return len(self.word2Index)

    def tag_size(self):
        return len(self.tags2Index)

    def char_size(self):
        return len(self.char2Index)

    def lookup_token(self, index):
        return self.index2Word.get(index, Const.UNK_TOKEN)

    def lookup_char(self, index):
        return self.char2Index.get(index, Const.UNK_ID)

    def vectorize(self, text, tags):

        token_seq = torch.LongTensor([self.lookup_index(token) for token in text])
        char_seq = torch.LongTensor([self.lookup_char(char) for char in text])
        tag_seq = torch.LongTensor([self.lookup_tag(tag) for tag in tags])

        token_pad = torch.cat([token_seq, token_seq.new_zeros(self.max_token_len - token_seq.size(0))], 0)
        char_pad = torch.cat([char_seq, char_seq.new_zeros(self.max_char_len - char_seq.size(0))], 0)
        tag_pad = torch.cat([tag_seq, tag_seq.new_zeros(self.max_token_len - tag_seq.size(0))], 0)

        return token_pad, char_pad, tag_pad



    def devectorize(self, tag_idx):
        return [self.index2tags[int(id)] for id in tag_idx]
