# models.py

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size=27, embed_size=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1, max_seq_length=19, vocab_index=None):
        nn.Module.__init__(self)
        LanguageModel.__init__(self)

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.vocab_index = vocab_index


        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))

        # transformer encoder
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # output layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        mask = self._generate_causal_mask(src.size(1))

        memory = self.transformer(src, mask=mask)
        out = self.fc_out(memory)
        log_probs = F.log_softmax(out, dim=-1)
        return log_probs

    def _generate_causal_mask(self, size):

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'

        # Create a causal mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # Upper triangular matrix
        mask = mask.to(torch.float) * float('-inf')
        # replace nan with 0
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        #mask = mask.unsqueeze(0).unsqueeze(0)
        return mask.to(device=device)


    def get_next_char_log_probs(self, context):

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'

        # if context is less than max_seq_length, pad with spaces at the beginning
        if len(context) < 19:
            context = ' ' * (19 - len(context)) + context

        # if context is greater than max_seq_length, take the last max_seq_length characters
        if len(context) > 19:
            context = context[-19:]


        self.eval()
        context_tensor = torch.tensor([self.vocab_index.objs_to_ints[c] for c in context]).unsqueeze(0).to(device)
        log_probs = self.forward(context_tensor).squeeze(0)[-1]
        return log_probs.detach().cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            log_prob_sum += log_probs[self.vocab_index.objs_to_ints[char]]
            context += char
        return log_prob_sum


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    # Initialize the model
    model = NeuralLanguageModel(vocab_index=vocab_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.NLLLoss()

    # Chunk and prepare the data
    sequence_length = 20  # Set the length of sequences to train on
    chunk_size = sequence_length - 1  # Input sequence size
    input_data = []
    target_data = []

    for i in range(len(train_text) - sequence_length):
        input_seq = train_text[i:i + chunk_size]
        target_seq = train_text[i + 1:i + sequence_length]
        input_data.append([vocab_index.objs_to_ints[c] for c in input_seq])
        target_data.append([vocab_index.objs_to_ints[c] for c in target_seq])

    # train on the first examples to validate the model
    #input_tensor = torch.tensor(input_data[:1]).to(device)
    #target_tensor = torch.tensor(target_data[:1]).to(device)

    input_tensor = torch.tensor(input_data).to(device)
    target_tensor = torch.tensor(target_data).to(device)

    model.train()



    num_epochs = 10
    batch_size = 100

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(input_tensor), batch_size):
            optimizer.zero_grad()
            batch_input = input_tensor[i:i + batch_size]
            batch_target = target_tensor[i:i + batch_size]

            log_probs = model(batch_input)
            loss = criterion(log_probs.view(-1, model.vocab_size), batch_target.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(input_tensor)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model
