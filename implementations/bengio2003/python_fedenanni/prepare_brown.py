"""Download and preprocess the Brown corpus as described in Bengio et al. (2003),
"A Neural Probabilistic Language Model" (JMLR 3:1137-1155).

Reference (Section 4, Experimental Results):

    "Comparative experiments were performed on the Brown corpus which is a stream
     of 1,181,041 words, from a large variety of English texts and books. The
     first 800,000 words were used for training, the following 200,000 for
     validation (model selection, weight decay, early stopping) and the remaining
     181,041 for testing. The number of different words is 47,578 (including
     punctuation, distinguishing between upper and lower case, and including the
     syntactic marks used to separate texts and paragraphs). Rare words with
     frequency <= 3 were merged into a single symbol, reducing the vocabulary
     size to |V| = 16,383."

This script:
  1. downloads the Brown corpus (via NLTK),
  2. checks that its size matches the count reported in the paper,
  3. tokenises it following the paper (keep case, 800k/200k/181,041 split,
     merge words with frequency <= 3 into a single <unk> symbol),
  4. saves the result to implementations/bengio2003/data.

Note on reproducibility: the paper's exact stream length (1,181,041) and vocab
size (16,383) depend on the precise source and tokenisation used in 2003. NLTK's
tokenisation of the Brown corpus yields ~1,161,192 tokens, so the size check
below WARNS on a mismatch and proceeds (this behaviour was chosen deliberately).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants from the paper (Section 4).
# ---------------------------------------------------------------------------
PAPER_TOTAL_WORDS = 1_181_041
PAPER_TRAIN_WORDS = 800_000
PAPER_VALID_WORDS = 200_000
PAPER_TEST_WORDS = 181_041  # 1,181,041 - 800,000 - 200,000
PAPER_NUM_DIFFERENT_WORDS = 47_578  # different words before rare-word merging
PAPER_VOCAB_SIZE = 16_383  # |V| after merging words with frequency <= 3

RARE_WORD_MAX_FREQ = 3  # "Rare words with frequency <= 3 were merged"
UNK_TOKEN = "<unk>"     # the "single symbol" rare words are merged into

# Output location: implementations/bengio2003/data (relative to this file).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Step 1: download the Brown corpus.
# ---------------------------------------------------------------------------
def download_brown() -> list[str]:
    """Download the Brown corpus via NLTK and return it as a flat token stream."""
    import nltk
    from nltk.corpus import brown

    try:
        brown.words()  # triggers LookupError if the data is not present yet
    except LookupError:
        print("Downloading the Brown corpus via NLTK ...")
        nltk.download("brown")

    tokens = list(brown.words())
    print(f"Downloaded Brown corpus: {len(tokens):,} tokens.")
    return tokens


# ---------------------------------------------------------------------------
# Step 2: check the size against the paper.
# ---------------------------------------------------------------------------
def check_size(tokens: list[str]) -> None:
    """Compare the token count to the paper's 1,181,041 and warn on mismatch."""
    actual = len(tokens)
    expected = PAPER_TOTAL_WORDS
    diff = actual - expected
    pct = 100.0 * diff / expected

    print("\n--- Size check (Bengio et al. 2003, Section 4) ---")
    print(f"  Expected (paper): {expected:,} words")
    print(f"  Actual (NLTK):    {actual:,} tokens")
    print(f"  Difference:       {diff:+,} ({pct:+.2f}%)")
    if actual == expected:
        print("  MATCH: token count matches the paper exactly.")
    else:
        print(
            "  WARNING: token count does NOT match the paper. This is expected "
            "with NLTK's tokenisation (it excludes some of the text/paragraph\n"
            "  separator marks the paper counted). Proceeding anyway."
        )
    print("-------------------------------------------------\n")


# ---------------------------------------------------------------------------
# Step 3: tokenise / preprocess as in the paper.
# ---------------------------------------------------------------------------
def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Build the vocabulary as in the paper.

    Frequencies are counted over the *whole* stream (the paper reports the number
    of different words over the full corpus). Words with frequency <= 3 are
    merged into a single <unk> symbol. Case is preserved (the paper distinguishes
    upper and lower case for the Brown corpus; only the AP News corpus was
    lowercased).
    """
    counts = Counter(tokens)
    num_different = len(counts)

    kept = sorted(w for w, c in counts.items() if c > RARE_WORD_MAX_FREQ)
    num_rare = num_different - len(kept)

    # Vocabulary: <unk> at index 0, then kept words in sorted order.
    vocab = {UNK_TOKEN: 0}
    for word in kept:
        vocab[word] = len(vocab)

    print("--- Vocabulary (rare-word merging) ---")
    print(f"  Different words (before merge): {num_different:,} "
          f"(paper: {PAPER_NUM_DIFFERENT_WORDS:,})")
    print(f"  Words with freq <= {RARE_WORD_MAX_FREQ} merged into {UNK_TOKEN!r}: "
          f"{num_rare:,}")
    print(f"  Resulting |V| (incl. {UNK_TOKEN}): {len(vocab):,} "
          f"(paper: {PAPER_VOCAB_SIZE:,})")
    print("--------------------------------------\n")
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Map each token to its vocabulary id, sending rare/OOV words to <unk>."""
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(tok, unk_id) for tok in tokens]


def split_stream(ids: list[int]) -> dict[str, list[int]]:
    """Split the encoded stream into train/valid/test by the paper's boundaries."""
    train_end = PAPER_TRAIN_WORDS
    valid_end = PAPER_TRAIN_WORDS + PAPER_VALID_WORDS

    # If the actual corpus is shorter/longer than the paper's, the split simply
    # uses whatever tokens are available (test = everything after validation).
    return {
        "train": ids[:train_end],
        "valid": ids[train_end:valid_end],
        "test": ids[valid_end:],
    }


# ---------------------------------------------------------------------------
# Step 4: save to implementations/bengio2003/data.
# ---------------------------------------------------------------------------
def save(vocab: dict[str, int], splits: dict[str, list[int]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Vocabulary: one token per line, line number (0-indexed) == token id.
    vocab_path = DATA_DIR / "vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as f:
        for word, _id in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(word + "\n")

    # Encoded splits: whitespace-separated integer ids.
    for name, ids in splits.items():
        path = DATA_DIR / f"{name}.ids"
        with path.open("w", encoding="utf-8") as f:
            f.write(" ".join(map(str, ids)))

    # Stats / provenance.
    stats = {
        "source": "NLTK brown.words()",
        "paper_total_words": PAPER_TOTAL_WORDS,
        "actual_total_tokens": sum(len(v) for v in splits.values()),
        "paper_vocab_size": PAPER_VOCAB_SIZE,
        "actual_vocab_size": len(vocab),
        "rare_word_max_freq": RARE_WORD_MAX_FREQ,
        "unk_token": UNK_TOKEN,
        "case_sensitive": True,
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "paper_split_sizes": {
            "train": PAPER_TRAIN_WORDS,
            "valid": PAPER_VALID_WORDS,
            "test": PAPER_TEST_WORDS,
        },
    }
    with (DATA_DIR / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {DATA_DIR}:")
    print(f"  vocab.txt  ({len(vocab):,} tokens)")
    for name, ids in splits.items():
        print(f"  {name}.ids   ({len(ids):,} tokens)")
    print("  stats.json")


def main() -> None:
    tokens = download_brown()
    check_size(tokens)
    vocab = build_vocab(tokens)
    ids = encode(tokens, vocab)
    splits = split_stream(ids)
    save(vocab, splits)


if __name__ == "__main__":
    main()
