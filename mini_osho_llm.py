"""
Mini Osho LLM with Byte Pair Encoding (BPE)
============================================
A modern, lightweight Transformer Language Model trained on Osho's
"The Book of Secrets" (Discourses on Vigyan Bhairav Tantra - 112 Meditation Techniques),
featuring:
  1. Automatic text extraction from 'The Book of Secrets.pdf' with caching
  2. Custom Byte Pair Encoding (BPE) subword tokenization
  3. RMSNorm (Root Mean Square Normalization)
  4. RoPE (Rotary Position Embeddings)
  5. GQA (Grouped-Query Attention with KV repeating)
  6. SwiGLU (Swish-Gated Linear Units)
  7. Weight tying between token embeddings and LM head
  8. Autoregressive text generation with temperature sampling
"""

import os
import re
import math
import time
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 1. Device Setup & Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DROPOUT = 0.2
NUM_BPE_MERGES = 500  # Number of BPE merge operations to learn

# ──────────────────────────────────────────────────────────────────────────────
# 2. Data Loading & Extraction from 'The Book of Secrets.pdf'
# ──────────────────────────────────────────────────────────────────────────────
PDF_PATH = "The Book of Secrets.pdf"
TXT_PATH = "book_of_secrets.txt"

if not os.path.exists(TXT_PATH):
    if os.path.exists(PDF_PATH):
        print(f"Extracting text from '{PDF_PATH}'...")
        import pypdf
        reader = pypdf.PdfReader(PDF_PATH)
        pages_text = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                pages_text.append(t)
        text = "\n\n".join(pages_text)
        with open(TXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {len(reader.pages)} pages ({len(text):,} characters) and cached to '{TXT_PATH}'")
    else:
        raise FileNotFoundError(f"Could not find '{PDF_PATH}' or '{TXT_PATH}' in {os.getcwd()}")
else:
    print(f"Loading cached text from '{TXT_PATH}'...")
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

print(f"Total characters: {len(text):,}")
print(f"First 250 characters:\n{text[:250]!r}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Byte Pair Encoding (BPE) Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Training BPE Tokenizer ({NUM_BPE_MERGES} merges on Osho's text)...")
print("=" * 60)

# Word splitting preserving whitespace
words = re.findall(r"\S+|\s+", text)
word_counts = Counter(words)

chars = sorted(set(text))
UNK_TOKEN = "<unk>"
base_tokens = [UNK_TOKEN] + chars

bpe_vocab = {tuple(w): count for w, count in word_counts.items()}

def get_pair_stats(vocab):
    pairs = Counter()
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pairs[(word_tuple[i], word_tuple[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    replacement = "".join(pair)
    for word_tuple, freq in vocab.items():
        new_tuple = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                new_tuple.append(replacement)
                i += 2
            else:
                new_tuple.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_tuple)] = freq
    return new_vocab

t_bpe_start = time.time()
merges = {}
for i in range(NUM_BPE_MERGES):
    pairs = get_pair_stats(bpe_vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    bpe_vocab = merge_vocab(best_pair, bpe_vocab)
    merges[best_pair] = "".join(best_pair)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  Merge {i+1:3d}: {best_pair[0]!r} + {best_pair[1]!r} -> {merges[best_pair]!r} (frequency: {pairs[best_pair]:,})")

print(f"BPE training complete in {time.time() - t_bpe_start:.2f}s!")

token_list = base_tokens + [t for t in merges.values() if t not in base_tokens]
token_to_id = {tok: i for i, tok in enumerate(token_list)}
id_to_token = {i: tok for i, tok in enumerate(token_list)}
vocab_size = len(token_list)

print(f"Final Vocabulary size: {vocab_size} ({len(base_tokens)} base + {len(merges)} merged)")

# Fast cached word encoder
word_cache = {}

def encode_word(w):
    if w in word_cache:
        return word_cache[w]
    tokens = list(w)
    for pair, replacement in merges.items():
        if len(tokens) < 2:
            break
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(replacement)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    unk_id = token_to_id[UNK_TOKEN]
    ids = [token_to_id.get(t, unk_id) for t in tokens]
    word_cache[w] = ids
    return ids

def encode(s):
    w_list = re.findall(r"\S+|\s+", s)
    out = []
    for w in w_list:
        out.extend(encode_word(w))
    return out

def decode(ids):
    return "".join(id_to_token.get(i, UNK_TOKEN) for i in ids)

# Demo encoding
demo_str = "Meditation is witnessing the breath."
enc_demo = encode(demo_str)
print(f"Demo encode: {demo_str!r} -> {enc_demo}")
print(f"Demo decode: {enc_demo} -> {decode(enc_demo)!r}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset Preparation & Batching
# ──────────────────────────────────────────────────────────────────────────────
print("\nEncoding full corpus with BPE tokenizer...")
t_enc = time.time()
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded {len(text):,} characters -> {len(data):,} BPE tokens in {time.time() - t_enc:.2f}s")
print(f"Compression ratio: {len(text) / len(data):.2f}x (avg {len(text) / len(data):.2f} chars/token)")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training tokens:   {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}\n")

def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i: i + context_length] for i in ix])
    y = torch.stack([d[i + 1: i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Architecture Components (RMSNorm, RoPE, GQA, SwiGLU)
# ──────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        b, seq, _ = x.shape
        q = self.q_proj(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=DROPOUT, training=self.training)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return F.dropout(self.w_down(gate * up), p=DROPOUT, training=self.training)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden_dim)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, ffn_hidden_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ──────────────────────────────────────────────────────────────────────────────
# 6. Model Initialization & Configuration
# ──────────────────────────────────────────────────────────────────────────────
config = {
    "vocab_size": vocab_size,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden_dim": 680,
    "max_seq_len": 256,
}

model = MiniLLM(**config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 55)
print("  MODEL SUMMARY (Mini Osho LLM + BPE)")
print("=" * 55)
print(f"  Corpus:          Osho - The Book of Secrets")
print(f"  Vocabulary:      {config['vocab_size']} (BPE tokens)")
print(f"  Embedding dim:   {config['d_model']}")
print(f"  Layers:          {config['n_layers']}")
print(f"  Query heads:     {config['n_heads']}")
print(f"  KV heads:        {config['n_kv_heads']} (GQA ratio: {config['n_heads']//config['n_kv_heads']}:1)")
print(f"  FFN hidden dim:  {config['ffn_hidden_dim']}")
print(f"  Context length:  {config['max_seq_len']} tokens (~{int(config['max_seq_len'] * (len(text)/len(data)))} characters)")
print(f"  Head dim:        {config['d_model'] // config['n_heads']}")
print("=" * 55)
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size (approx):  {total_params * 4 / 1e6:.1f} MB (float32)")
print("=" * 55)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Training Setup
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
CONTEXT_LEN = config["max_seq_len"]
LEARNING_RATE = 3e-4
MAX_STEPS = 3000
EVAL_INTERVAL = 250
EVAL_STEPS = 20
LOG_INTERVAL = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb = get_batch(split, BATCH_SIZE, CONTEXT_LEN)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 8. Autoregressive Generation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, prompt, max_new_tokens=250, temperature=0.8):
    """
    Autoregressively generate text from a prompt using BPE tokens.
    """
    model.eval()
    tokens = encode(prompt)
    if len(tokens) == 0:
        tokens = [0]
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -config["max_seq_len"]:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

def load_checkpoint(checkpoint_path="mini_osho_llm_checkpoint.pt", device=device):
    """
    Load saved model weights and tokenizer mapping from disk.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist!")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_model = MiniLLM(**ckpt["config"]).to(device)
    saved_model.load_state_dict(ckpt["model_state_dict"])
    saved_model.eval()
    print(f"Loaded checkpoint from '{checkpoint_path}' successfully!")
    return saved_model, ckpt

# ──────────────────────────────────────────────────────────────────────────────
# 9. Main Execution (Training & Generation)
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate():
    print("\nStarting training ....")
    print(f"   {MAX_STEPS} steps, batch_size={BATCH_SIZE}, context_len={CONTEXT_LEN}")
    print(f"   Random guessing loss baseline: ln({vocab_size}) = {math.log(vocab_size):.2f}")
    print("-" * 60)

    train_losses = []
    val_losses = []
    step_log = []
    start_time = time.time()

    model.train()
    for step in range(MAX_STEPS):
        xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LEN)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f" Steps {step:5d}/{MAX_STEPS} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss()
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            step_log.append(step)
            if step > 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                remaining = (MAX_STEPS - step) / steps_per_sec
                print(f" >>> Eval @ step {step}: train={losses['train']:.4f}, val={losses['val']:.4f} | ~{remaining:.0f}s remaining")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete! Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss:   {val_losses[-1]:.4f}")

    # Save trained model and tokenizer checkpoint
    checkpoint_file = "mini_osho_llm_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "merges": merges,
        "base_tokens": base_tokens,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }, checkpoint_file)
    print(f"Model and tokenizer checkpoint saved to '{checkpoint_file}' ({os.path.getsize(checkpoint_file) / 1e6:.1f} MB)")

    # Plot loss curve
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(step_log, train_losses, "b-o", label="Train", markersize=4)
        ax.plot(step_log, val_losses, "r-o", label="Validation", markersize=4)
        ax.axhline(y=math.log(vocab_size), color="gray", linestyle="--", alpha=0.5,
                   label=f"Random guessing ({math.log(vocab_size):.2f})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress (Mini Osho LLM + BPE Tokenizer)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("mini_osho_llm_loss.png")
        print("Loss plot saved to mini_osho_llm_loss.png")
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Generation Tests
    print("\n" + "=" * 60)
    print("TEXT GENERATION DEMOS (Osho - The Book of Secrets)")
    print("=" * 60)

    prompt = "DEVI ASKS:\nOH SHIVA,"
    for temp in [0.5, 0.8, 1.0]:
        print(f"\n{'-' * 40}\nTemperature = {temp}\n{'-' * 40}")
        out = generate(model, prompt, max_new_tokens=200, temperature=temp)
        print(out)

    sample_prompts = [
        "OSHO:\nWhat is the secret of meditation?",
        "When the breath is turned inward,",
        "The technique of witnessing is",
        "Mind is not an entity; it is",
    ]
    for p in sample_prompts:
        print(f"\n{'=' * 50}\nPrompt: {p!r}\n{'=' * 50}")
        print(generate(model, p, max_new_tokens=150, temperature=0.8))

if __name__ == "__main__":
    train_and_evaluate()
