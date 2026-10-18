# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
Byte Pair Encoding (BPE) Tokenizer for Shakespeare Dataset
==========================================================

This file implements BPE tokenization as an alternative to the
character-level tokenization used in Untitled-1.ipynb.

HOW IT DIFFERS FROM THE CURRENT CODE:
--------------------------------------
Current code (Untitled-1.ipynb):
  - Tokenizes at the CHARACTER level
  - vocab_size = 65 (just the unique characters in Shakespeare)
  - Each token = 1 character (e.g. 'hello' → [20, 43, 50, 50, 53])
  - 1,115,394 characters → 1,115,394 tokens (ratio 1:1)

BPE (this file):
  - Starts with characters, then iteratively MERGES the most frequent
    adjacent pairs into new tokens (subwords/words)
  - vocab_size grows with the number of merge operations you allow
  - Each token can represent multiple characters (e.g. ' the' → single token)
  - Same text becomes FEWER tokens (better compression)
  - More meaningful tokens → model can learn language patterns faster
"""

import os
import re
from collections import Counter, defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load the text
# ──────────────────────────────────────────────────────────────────────────────

with open("shakespeare.txt") as f:
    text = f.read()

print(f"Total characters: {len(text):,}")
print(f"First 200 characters:\n{text[:200]}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Current code recap (character-level)
# ──────────────────────────────────────────────────────────────────────────────

chars = sorted(set(text))
char_vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def char_encode(s):
    return [char_to_idx[c] for c in s]

def char_decode(ids):
    return "".join([idx_to_char[i] for i in ids])

sample = "Hello world"
print("=" * 60)
print("CHARACTER-LEVEL (current code in Untitled-1.ipynb)")
print("=" * 60)
print(f"  Vocabulary size : {char_vocab_size}")
print(f"  Tokens in text  : {len(text):,}  (same as characters - ratio 1:1)")
print(f"  Example encode  : {sample!r} -> {char_encode(sample)}")
print(f"  Example decode  : {char_encode(sample)} -> {char_decode(char_encode(sample))!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. BPE implementation
# ──────────────────────────────────────────────────────────────────────────────

def build_vocab_from_text(text: str) -> dict:
    """
    Split text into words (preserving spaces as part of the token,
    GPT-2 style) and count each word's character-sequence frequency.

    Returns a dict: { tuple_of_chars : frequency }
    e.g. { (' ', 't', 'h', 'e') : 12345 }
    """
    # Tokenise into 'words' — keep leading space as part of the word
    # so that " the" and "the" are treated as different units.
    word_counts = Counter(re.findall(r"\S+|\s+", text))
    vocab = {}
    for word, freq in word_counts.items():
        # Represent each word as a tuple of individual characters
        token_seq = tuple(word)
        vocab[token_seq] = vocab.get(token_seq, 0) + freq
    return vocab


def get_pair_stats(vocab: dict) -> Counter:
    """Count how often each adjacent pair of symbols appears."""
    pairs = Counter()
    for token_seq, freq in vocab.items():
        for a, b in zip(token_seq, token_seq[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(vocab: dict, pair: tuple) -> dict:
    """Replace every occurrence of `pair` in vocab with a merged symbol."""
    merged = "".join(pair)
    new_vocab = {}
    for token_seq, freq in vocab.items():
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == pair:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_vocab[tuple(new_seq)] = freq
    return new_vocab


def train_bpe(text: str, num_merges: int = 500):
    """
    Train BPE on text for `num_merges` merge operations.

    Returns:
        merges  : list of (pair, merged_token) in order applied
        vocab   : final {symbol_seq: freq} after all merges
        bpe_vocab: set of all tokens (initial chars + merged)
    """
    vocab = build_vocab_from_text(text)

    # Initial symbol set = all unique characters
    all_symbols = set(c for token_seq in vocab for c in token_seq)

    merges = []
    print(f"\nRunning {num_merges} BPE merge operations...")
    for i in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merged = "".join(best_pair)
        vocab = merge_pair(vocab, best_pair)
        merges.append((best_pair, merged))
        all_symbols.add(merged)

        if (i + 1) % 100 == 0:
            print(f"  Merge {i+1:4d}: {best_pair[0]!r} + {best_pair[1]!r} -> {merged!r}  "
                  f"(occurred {pairs[best_pair]:,}x)")

    return merges, vocab, all_symbols


# ──────────────────────────────────────────────────────────────────────────────
# 4. Build the BPE encode / decode functions
# ──────────────────────────────────────────────────────────────────────────────

def build_bpe_encoder(merges: list, initial_chars: set):
    """
    Build a simple greedy BPE encoder from the list of merge rules.

    The encoder applies merge rules in order (same as training) to
    a raw string and returns a list of integer token ids.
    """
    # token -> id  (sorted so deterministic)
    all_tokens = sorted(initial_chars) + [m for _, m in merges
                                           if m not in initial_chars]
    token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    # Build a fast lookup: pair -> merged string
    merge_lookup = {pair: merged for pair, merged in merges}

    def encode(s: str) -> list:
        # Start: every character is its own token
        tokens = list(s)
        # Apply merge rules greedily (left-to-right, in training order)
        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in merge_lookup:
                        new_tokens.append(merge_lookup[pair])
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens
        return [token_to_id[t] for t in tokens]

    def decode(ids: list) -> str:
        return "".join(id_to_token[i] for i in ids)

    return encode, decode, token_to_id, id_to_token


# ──────────────────────────────────────────────────────────────────────────────
# 5. Run BPE and compare
# ──────────────────────────────────────────────────────────────────────────────

NUM_MERGES = 500   # change this to get a larger / smaller BPE vocabulary

merges, final_vocab, all_symbols = train_bpe(text, num_merges=NUM_MERGES)

# Extract the initial character set (before any merges)
initial_chars = set(chars)  # same 65 chars as the notebook

bpe_encode, bpe_decode, token_to_id, id_to_token = build_bpe_encoder(
    merges, initial_chars
)

# Encode the entire Shakespeare corpus with BPE
print("\nEncoding full text with BPE (this may take a moment)...")
bpe_ids = bpe_encode(text)

bpe_vocab_size = len(token_to_id)

print("\n" + "=" * 60)
print("BYTE PAIR ENCODING (this file)")
print("=" * 60)
print(f"  Vocabulary size  : {bpe_vocab_size}  "
      f"(= {char_vocab_size} base chars + {NUM_MERGES} merges)")
print(f"  Tokens in text   : {len(bpe_ids):,}")
print(f"  Compression ratio: {len(text) / len(bpe_ids):.2f}x  "
      f"(1 BPE token = {len(text) / len(bpe_ids):.2f} chars on average)")

sample = "Hello world"
encoded = bpe_encode(sample)
decoded = bpe_decode(encoded)
print(f"\n  Example encode   : {sample!r} -> {encoded}")
print(f"  Example decode   : {encoded} -> {decoded!r}")

# Show a few of the learnt subword tokens
print(f"\n  First 20 merged tokens (after base chars):")
merged_tokens = [m for _, m in merges[:20]]
for i, tok in enumerate(merged_tokens, 1):
    print(f"    {i:3d}. {tok!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Side-by-side comparison summary
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
rows = [
    ("Property",               "Character-level (notebook)", f"BPE ({NUM_MERGES} merges)"),
    ("-" * 22,                 "-" * 28,                      "-" * 24),
    ("Vocab size",             str(char_vocab_size),           str(bpe_vocab_size)),
    ("Token count",            f"{len(text):,}",               f"{len(bpe_ids):,}"),
    ("Compression",            "1.00x",                        f"{len(text)/len(bpe_ids):.2f}x"),
    ("Token granularity",      "1 character",                  "1-N characters (subwords)"),
    ("Rare words handled?",    "Yes (char by char)",           "Yes (falls back to chars)"),
    ("Context window usage",   "Longer sequences needed",      "Shorter sequences for same text"),
    ("Training data needed",   "None (pure lookup)",           f"Corpus scan for {NUM_MERGES} merges"),
]
for row in rows:
    print(f"  {row[0]:<24} {row[1]:<30} {row[2]}")

print("""
KEY INSIGHTS
-------------------------------------------------------------
1. VOCABULARY SIZE
   - Char-level: 65 tokens (just the characters that appear).
   - BPE: 65 + NUM_MERGES tokens -- you control the trade-off.

2. SEQUENCE LENGTH (= context window pressure)
   - Char-level: 1 token per character -> very long sequences.
   - BPE: common words/subwords become single tokens -> shorter
     sequences for the same text -> the transformer can "see"
     more text within the same context window.

3. SEMANTICS
   - Char-level tokens carry no inherent meaning.
   - BPE tokens like ' the', ' and', 'ing' carry subword meaning,
     giving the model a head-start on morphology & common words.

4. TRAINING COST
   - Char-level: zero preprocessing cost.
   - BPE: O(corpus x merges) preprocessing, but the resulting
     model trains faster due to shorter sequences.

5. REAL-WORLD USE
   - GPT-2 / GPT-4 use BPE with ~50,000 merges.
   - LLaMA uses SentencePiece (similar idea) with 32,000 tokens.
   - The notebook's 65-token char-level approach is great for
     teaching but would be impractical at production scale.
""")
