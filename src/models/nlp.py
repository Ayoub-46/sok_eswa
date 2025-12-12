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
    
    Architecture:
    1. Embedding (Pre-trained GloVe, frozen)
    2. LSTM (2 Layers, 256 Hidden)
    3. Linear Head (Binary Classification)
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True) # Expects [Batch, Seq, Feature]
        
        # 3. Fully Connected Layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

    def load_pretrained_embeddings(self, embeddings, freeze=True):
        """
        Loads GloVe vectors and optionally freezes them.
        
        In FL, freezing is CRITICAL. 
        If False: The model update size is ~20MB (50k * 100 floats).
        If True: The model update size is ~1MB (LSTM weights only).
        """
        # Handle size mismatch if vocab changed
        if embeddings.shape != self.embedding.weight.shape:
            print(f"Resize embedding: {self.embedding.weight.shape} -> {embeddings.shape}")
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=0)
        else:
            self.embedding.weight.data.copy_(embeddings)
            if freeze:
                self.embedding.weight.requires_grad = False
        
        state = "Frozen" if freeze else "Trainable"
        print(f"Embeddings loaded and {state}.")

    def forward(self, text):
        # text shape: [batch size, seq_len]
        
        # 1. Embed
        # [batch, seq_len, emb_dim]
        embedded = self.dropout(self.embedding(text))
        
        # 2. LSTM
        # output: [batch, seq_len, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # 3. Extract final state
        # We take the hidden state of the last layer
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward states
            hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # Take the last layer state
            hidden_final = hidden[-1,:,:]
            
        # Apply dropout to the hidden state before the FC layer
        hidden_final = self.dropout(hidden_final)
            
        # 4. Prediction (Logits)
        # We return logits because we use BCEWithLogitsLoss for numerical stability
        prediction = self.fc(hidden_final)
        
        return prediction