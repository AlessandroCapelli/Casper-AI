import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 16  # How Many Independent Sequences Will We Process in Parallel?
block_size = 32  # The Maximum Context Length for Predictions
max_iters = 5000  # The Maximum Number of Training Iterations
eval_interval = 100  # How Often (in Iterations) to Evaluate the Loss
learning_rate = 1e-3  # The Learning Rate for the Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set Device to CUDA if Available, Else CPU
eval_iters = 200  # The Number of Iterations for Loss Evaluation
n_embd = 64  # Embedding Dimension Size
n_head = 4  # Number of Attention Heads in the Multi-Head Attention
n_layer = 4  # Number of Transformer Blocks (Layers)
dropout = 0.0  # Dropout Rate
seed = 1337  # Seed for Reproducibility

torch.manual_seed(seed)  # Set the Random Seed for Torch

# Download the Dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract Unique Characters
# Here Are All the Unique Characters That Occur in the Text.
chars = sorted(list(set(text)))
vocab_size = len(chars)  # Determine the Vocabulary Size

# Create Character to Integer Mapping
# Create a Mapping from Characters to Integers. This Is a Simple Example of Tokenisation.
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # Encoder: Take a String, Output a List of Integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: Take a List of Integers, Output a String

# Split Data into Training and Validation Sets
# Convert the Entire Text to a Tensor of Integers and Split It (90% for Training, 10% for Validation)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # First 90% Will Be Used for Training
train_data = data[:n]
val_data = data[n:]

# Data Loading Function
def get_batch(split):
    """ The function get_batch randomly selects starting indices from the training or 
    	validation data and extracts sequences of fixed length (block_size).
    	The targets are the input sequences shifted one position ahead. 
    	Both inputs and targets are moved to the specified computation device.
    """
    
    # Generate a Small Batch of Data of Inputs x and Targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # Sample Random Start Indices for the Batch
    x = torch.stack([data[i:i+block_size] for i in ix])  # Create Input Sequences of Length block_size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # Create Target Sequences Shifted by One
    x, y = x.to(device), y.to(device)  # Move the Batch to the Specified Device
    return x, y

@torch.no_grad()
def estimate_loss():
    """ The function estimate_loss switches the model to evaluation mode 
    	(to disable dropout and other training-specific layers) and computes 
    	the average loss over a number of batches for both training and validation sets. 
    	It then returns these averaged losses.
    """
    
    # Evaluate the Loss on the Train and Validation Sets Without Updating the Model
    out = {}
    model.eval()  # Set the Model to Evaluation Mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Get a Batch of Data
            logits, loss = model(X, Y)  # Get the Logits and Loss from the Model
            losses[k] = loss.item()  # Record the Loss
        out[split] = losses.mean()  # Compute the Mean Loss for the Split
    model.train()  # Set the Model Back to Training Mode
    return out

class Head(nn.Module):
    """ One Head of Self-Attention
		The Head class implements a single self-attention head. It computes key, 
        query, and value projections, applies a scaled dot-product attention mechanism 
        with causal masking (to prevent attending to future tokens), and returns the 
        aggregated output.
    """

    def __init__(self, head_size):
        super().__init__()
        # Initialize Linear Layers for Key, Query, and Value Projections Without Bias.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register a Lower Triangular Matrix for Causal Masking.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Get Batch Size, Sequence Length, and Embedding Dimension
        k = self.key(x)   # Compute Key Projections; Shape: (B, T, Head_Size)
        q = self.query(x) # Compute Query Projections; Shape: (B, T, Head_Size)
        # Compute Attention Scores (Affinities) and Scale Them.
        wei = q @ k.transpose(-2, -1) * C**-0.5  # Shape: (B, T, T)
        # Apply Causal Masking to Prevent Access to Future Tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # Convert Scores to Probabilities Using Softmax
        wei = self.dropout(wei)
        v = self.value(x)  # Compute Value Projections; Shape: (B, T, Head_Size)
        out = wei @ v  # Weighted Sum of Values; Shape: (B, T, Head_Size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Heads of Self-Attention in Parallel 
		The MultiHeadAttention class runs several self-attention heads in parallel, 
        concatenates their outputs, and then projects them back to the original 
        embedding dimension.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a ModuleList of Independent Attention Heads.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # Final Linear Projection to Combine the Heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the Outputs of All Heads Along the Last Dimension.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # Apply the Projection and Dropout
        return out

class FeedFoward(nn.Module):
    """ A Simple Linear Layer Followed by a Non-Linearity 
		The FeedFoward class implements a simple feedforward neural network that 
        increases the dimensionality temporarily, applies a ReLU non-linearity, 
        and then reduces the dimensionality back to the original embedding size.
    """

    def __init__(self, n_embd):
        super().__init__()
        # Create a Two-Layer Feedforward Network with a ReLU Activation and Dropout.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication Followed by Computation 
		The Block class represents a single Transformer block. It applies multi-head 
        self-attention and a feedforward network, each preceded by layer normalization, 
        and uses residual connections to help with gradient flow.
    """

    def __init__(self, n_embd, n_head):
        # n_embd: Embedding Dimension, n_head: Number of Attention Heads
        super().__init__()
        head_size = n_embd // n_head  # Determine the Size of Each Attention Head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-Attention Layer
        self.ffwd = FeedFoward(n_embd)  # FeedForward Layer
        self.ln1 = nn.LayerNorm(n_embd)  # First Layer Normalization
        self.ln2 = nn.LayerNorm(n_embd)  # Second Layer Normalization

    def forward(self, x):
        # Apply Self-Attention on the Normalized Input and Add a Residual Connection.
        x = x + self.sa(self.ln1(x))
        # Apply the FeedForward Network on the Normalized Output and Add a Residual Connection.
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """ Super Simple Bigram Language Model Enhanced with Transformer Blocks 
		The BigramLanguageModel class builds a language model that uses both token 
        and positional embeddings. It passes the embeddings through several Transformer 
        blocks, applies a final normalization, and then uses a linear layer to produce 
        logits over the vocabulary. In training mode, if targets are provided, it computes 
        the cross-entropy loss. The generate method allows the model to generate new text
        tokens iteratively given a starting context.
    """

    def __init__(self):
        super().__init__()
        # Create Token and Position Embedding Tables.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack Multiple Transformer Blocks.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final Layer Normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Linear Layer to Produce Logits for Each Token

    def forward(self, idx, targets=None):
        B, T = idx.shape  # Batch Size and Sequence Length

        # Get Token Embeddings and Add Position Embeddings.
        tok_emb = self.token_embedding_table(idx)  # Shape: (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Shape: (T, n_embd)
        x = tok_emb + pos_emb  # Combine the Embeddings; Shape: (B, T, n_embd)
        x = self.blocks(x)  # Pass Through Transformer Blocks; Shape: (B, T, n_embd)
        x = self.ln_f(x)  # Apply Final Layer Normalization; Shape: (B, T, n_embd)
        logits = self.lm_head(x)  # Compute Logits for Each Token; Shape: (B, T, vocab_size)

        # If Targets Are Provided, Compute the Cross-Entropy Loss.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate New Tokens Given an Initial Context.
        for _ in range(max_new_tokens):
            # Crop the Context to the Last block_size Tokens.
            idx_cond = idx[:, -block_size:]
            # Get Predictions for the Current Context.
            logits, loss = self(idx_cond)
            # Focus Only on the Last Time Step.
            logits = logits[:, -1, :]  # Shape: (B, vocab_size)
            # Apply Softmax to Convert Logits to Probabilities.
            probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)
            # Sample a Token from the Probability Distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)
            # Append the Sampled Token to the Existing Sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (B, T+1)
        return idx

""" Model Initialization, Training Loop, and Text Generation
	The model is instantiated and moved to the appropriate device. 
    The number of parameters is printed for reference. An AdamW optimizer 
    is created to update the model's parameters. A training loop runs for
    a fixed number of iterations, periodically evaluating and printing the 
    loss on both training and validation sets. After training, the model 
    generates text by iteratively sampling new tokens from its predictions, 
    starting with an initial context. The generated sequence is then decoded 
    into human-readable text and printed.
"""
model = BigramLanguageModel()  # Create an Instance of the Bigram Language Model
m = model.to(device)  # Move the Model to the Specified Device
# Print the Number of Parameters in the Model (in Millions)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# Create a PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Evaluate the Loss on Train and Validation Sets at Regular Intervals
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Sample a Batch of Training Data
    xb, yb = get_batch('train')

    # Perform a Forward Pass and Compute the Loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # Reset Gradients
    loss.backward()  # Backpropagate the Loss
    optimizer.step()  # Update the Model Parameters

# Generate Text from the Model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting Context (A Tensor of Zeros)
generated_indices = m.generate(context, max_new_tokens=2000)  # Generate New Tokens
print(decode(generated_indices[0].tolist()))  # Decode and Print the Generated Text
