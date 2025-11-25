import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 8, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        # Reshape for CrossEntropyLoss: [Batch, Vocab, Seq_Len]
        return logits.permute(0, 2, 1)