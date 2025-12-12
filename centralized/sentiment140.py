import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = 'data/sentiment140/training.1600000.processed.noemoticon.csv' # Update this path
GLOVE_PATH = 'data/glove/glove.6B.100d.txt'
SAMPLE_SIZE = 50000      # Use a subset for faster debugging (set to None for full data)
MAX_VOCAB_SIZE = 5000    # Maximum number of words in the vocabulary
MAX_SEQ_LEN = 100        # Max length of a tweet (padding/truncating)
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on device: {DEVICE}")

# --- 1. Data Preprocessing ---

def clean_text(text):
    """
    Standard tweet cleaning: remove URLs, user mentions, punctuation.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove user mentions
    text = re.sub(r'\d+', '', text)  # Remove numbers
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def load_and_preprocess_data(path, sample_size=None):
    print("Loading data...")
    # Sentiment140 has no headers. Columns: target, ids, date, flag, user, text
    cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
    try:
        df = pd.read_csv(path, encoding='latin-1', names=cols)
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please download Sentiment140.")
        return None, None

    # Map target: 0=Negative, 4=Positive. We want 0 and 1.
    df['target'] = df['target'].replace(4, 1)
    
    # Subsample for speed if requested
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Simple whitespace tokenization
    texts = [text.split() for text in df['clean_text']]
    labels = df['target'].values
    
    return texts, labels

# --- 2. Tokenization & Vocabulary Building ---

class Vocabulary:
    def __init__(self, max_size):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()

    def build_vocab(self, texts):
        for text in texts:
            self.word_counts.update(text)
        
        # Keep most common words
        most_common = self.word_counts.most_common(self.max_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def numericalize(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text]

# --- 3. PyTorch Dataset ---

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # Convert to indices
        indices = self.vocab.numericalize(text)
        
        # Padding / Truncating
        if len(indices) < self.max_len:
            # Pad
            indices += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(indices))
        else:
            # Truncate
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --- 4. Model Architecture (LSTM) ---

def load_glove_embeddings(vocab, glove_path, embedding_dim):
    """
    Creates an embedding matrix for our specific vocabulary using GloVe.
    """
    print(f"Loading GloVe vectors from {glove_path}...")
    embeddings_index = {}
    
    # 1. Parse the GloVe text file
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"Found {len(embeddings_index)} word vectors in GloVe.")

    # 2. Create the matrix for our vocab
    # Initialize with random noise or zeros
    embedding_matrix = np.zeros((len(vocab.word2idx), embedding_dim))
    
    hits = 0
    misses = 0
    
    for word, idx in vocab.word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Word found in GloVe
            embedding_matrix[idx] = embedding_vector
            hits += 1
        else:
            # Word not found (OOV). Leave as zeros or random initialization.
            # Usually, we initialize <UNK> and <PAD> specifically or leave them.
            if word == "<UNK>":
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))
            misses += 1

    print(f"Embeddings loaded. Hits: {hits}, Misses: {misses}")
    return torch.tensor(embedding_matrix, dtype=torch.float)

class SentimentLSTM_GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, output_dim=1):
        super(SentimentLSTM_GloVe, self).__init__()
        
        # 1. Initialize Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Load Pre-trained weights if provided
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({'weight': pretrained_embeddings})
            
            # OPTION A: Freeze embeddings (Faster, less bandwidth in FL, good for small data)
            # self.embedding.weight.requires_grad = False 
            
            # OPTION B: Fine-tune embeddings (Better accuracy, but updates huge matrix)
            self.embedding.weight.requires_grad = True 

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dense_out = self.dropout(hidden_cat)
        out = self.fc(dense_out)
        return self.sigmoid(out)

    
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Bidirectional doubles the hidden dimension output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        
        # LSTM output: [batch_size, seq_len, hidden_dim * 2]
        # hidden/cell: [num_layers * num_directions, batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        # We take the final hidden state. 
        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        dense_out = self.dropout(hidden_cat)
        out = self.fc(dense_out)
        return self.sigmoid(out)

# --- 5. Training & Evaluation Functions ---

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in tqdm(loader, desc="Training"):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted_classes = (predictions > 0.5).float()
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            preds = (predictions > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
    
    return total_loss / len(loader), accuracy, report

# --- 6. Main Execution ---

def main():
    # A. Load Data
    raw_texts, raw_labels = load_and_preprocess_data(DATA_PATH, SAMPLE_SIZE)
    if raw_texts is None: return

    # B. Split Data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        raw_texts, raw_labels, test_size=0.2, random_state=42, stratify=raw_labels
    )

    # C. Build Vocab
    print("Building Vocabulary...")
    vocab = Vocabulary(MAX_VOCAB_SIZE)
    vocab.build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab.word2idx)}")

    # D. create Datasets and Loaders
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, MAX_SEQ_LEN)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # E. Initialize Model
    embedding_matrix = load_glove_embeddings(vocab, GLOVE_PATH, EMBEDDING_DIM)
    model = SentimentLSTM_GloVe(
        vocab_size=len(vocab.word2idx), 
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=embedding_matrix # Pass the matrix here
    ).to(DEVICE)
    # model = SentimentLSTM(len(vocab.word2idx), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # F. Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _ = evaluate(model, test_loader, criterion)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # G. Final Evaluation
    print("\nFinal Evaluation on Test Set:")
    test_loss, test_acc, report = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    print(report)

    # H. Save for FL
    # In FL, you would send this state_dict to clients
    torch.save(model.state_dict(), "centralized_sentiment_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()