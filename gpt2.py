import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------
# hyperparameters
# -----------------
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 128  # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# ------------
# data loading
# ------------

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """
    Generate a small batch of data of inputs x and targets y.
    x, y have shape (B, T)
    """
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate train/val loss over a few batches."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -------------------------
# Transformer model pieces
# -------------------------


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)

        # causal mask (no looking ahead)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # softmax + dropout
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate along channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # final linear projection
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """
    A simple MLP: linear -> ReLU -> linear, with dropout.
    (Class name keeps the original lecture's typo.)
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: self-attn followed by feedforward, with pre-norm."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # pre-norm + residual
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets: (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        # idx: (B, T) current context
        for _ in range(max_new_tokens):
            # crop to last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get logits for current context
            logits, _ = self(idx_cond)
            # take last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # softmax to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# -------------------
# Training entrypoint
# -------------------


if __name__ == "__main__":
    model = GPTLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # periodically evaluate train/val loss
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        # get a batch
        xb, yb = get_batch("train")

        # forward, backward, update
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save trained model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "gpt_shakespeare.pt")

    # generate some text to inspect the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))