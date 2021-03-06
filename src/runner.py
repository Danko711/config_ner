from catalyst import dl
import torch
import pickle


class CustomRunner(dl.Runner):

    with open('./vect.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    vectorizer = vectorizer



    def _handle_batch(self, batch):
        features, tags = batch
        sents, chars = features

        self.model.train()

        #sents, chars, tags = sents.to(device), chars.to(device), tags.to(device)

        seq = self.model(sents, chars)  # , mask)
        seq_tens = [torch.Tensor(s) for s in seq]
        seq = torch.nn.utils.rnn.pad_sequence(seq_tens, batch_first=True).cpu().numpy()
        seq = torch.Tensor(seq)

        total_preds = [self.vectorizer.devectorize(i) for i in seq]
        total_tags = [self.vectorizer.devectorize(i) for i in tags]

        crf_nll = self.model.loss(sents, chars, tags)
        logits = self.model._lstm(sents, chars)

        self.input = {'x': sents, 'x_char': chars, 'targets': tags, 'total_tags': total_tags}  # 'mask': mask,
        self.output = {'preds': total_preds, 'crf_nll': self.model.loss, 'logits': logits}