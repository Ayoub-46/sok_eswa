import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 8, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # Padding index 0 is crucial for the embedding layer to ignore <PAD>
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        return logits.permute(0, 2, 1)
    

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, num_layers=2, pretrained_embeddings=None):
        super().__init__()
        
        # Load weights if provided, otherwise random
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)
    
class SentimentLSTM(nn.Module):
    """
    Standard 2-Layer LSTM for Federated Sentiment Analysis.
    Reference: Bagdasaryan et al. "How To Backdoor Federated Learning"
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim=100, 
                 hidden_dim=256, 
                 output_dim=1, 
                 n_layers=2, 
                 bidirectional=False, 
                 dropout=0.5, 
                 pad_idx=0):
        super(SentimentLSTM, self).__init__()
        
        # 1. Embedding Layer
        # We specify padding_idx so the model ignores the padding token (0) during training
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. LSTM Layer
        # batch_first=True expects input shape: [batch_size, seq_len]
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        
        # 3. Fully Connected Layer
        # If bidirectional, the hidden state size doubles
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # 5. Activation
        # BCEWithLogitsLoss includes Sigmoid, but we include it here if you use BCELoss
        self.sigmoid = nn.Sigmoid()

    def load_pretrained_embeddings(self, embeddings, freeze=True):
        """
        Helper to load GloVe vectors.
        Arguments:
            embeddings (torch.Tensor): Tensor of shape (vocab_size, embedding_dim)
            freeze (bool): If True, gradients won't update the embeddings (Standard FL practice)
        """
        self.embedding.weight.data.copy_(embeddings)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, text, text_lengths=None):
        # text shape: [batch size, sent len]
        
        # 1. Embed
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch size, sent len, emb dim]
        
        # 2. LSTM
        # We handle packing if lengths are provided (optimization for variable length sequences)
        if text_lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            # Unpack isn't strictly necessary if we only care about the final hidden state
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # 3. Extract final hidden state
        # hidden shape: [num layers * num directions, batch size, hid dim]
        if self.lstm.bidirectional:
            # Concat the final forward and backward hidden states
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # Take the last layer's hidden state
            hidden = self.dropout(hidden[-1,:,:])
            
        # 4. Prediction
        prediction = self.fc(hidden)
        
        # Optional: Apply sigmoid if your loss function doesn't do it automatically
        # prediction = self.sigmoid(prediction)
        
        return prediction