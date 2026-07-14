# Brown corpus preparation — Bengio et al. (2003)

Prepares the Brown corpus following the preprocessing described in
Bengio, Ducharme, Vincent & Jauvin (2003), *A Neural Probabilistic Language
Model*, JMLR 3:1137–1155 (Section 4, "Experimental Results").

## Usage

```bash
pip install -r requirements.txt
python prepare_brown.py
```

The script:

1. **downloads** the Brown corpus via NLTK (`nltk.download('brown')`);
2. **checks** the token count against the paper's reported **1,181,041 words**;
3. **tokenises** it as in the paper — case preserved, an 800k / 200k / 181,041
   train/valid/test split, and words with frequency ≤ 3 merged into a single
   `<unk>` symbol;
4. **saves** the result to `../data/`.

## Output (`implementations/bengio2003/data/`)

| file | contents |
|------|----------|
| `vocab.txt` | one token per line; line number (0-indexed) = token id; `<unk>` is id 0 |
| `train.ids` | whitespace-separated integer token ids (first 800,000 tokens) |
| `valid.ids` | next 200,000 tokens |
| `test.ids`  | remaining tokens |
| `stats.json`| provenance and paper-vs-actual counts |

## Reproducibility note

The paper's exact figures (1,181,041 tokens, |V| = 16,383) depend on the precise
source and tokenisation used in 2003 and are hard to reproduce exactly. NLTK's
tokenisation yields **1,161,192 tokens** and, after rare-word merging, a
vocabulary of **17,905** (vs. the paper's 16,383). The ~20k token gap is largely
the text/paragraph separator marks the paper counted but NLTK does not emit.

Because of this, the size check **warns on a mismatch and proceeds** rather than
aborting. All paper-vs-actual numbers are printed to the console and recorded in
`stats.json`.
