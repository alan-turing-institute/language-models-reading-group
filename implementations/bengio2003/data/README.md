# Data source and license

The files in this directory (`vocab.txt`, `*.ids`, `stats.json`) are a derived
form of the **Brown Corpus**, preprocessed as described in Bengio et al. (2003).
They are **not** covered by this repository's own LICENSE; the corpus terms below
apply to the data.

## Corpus

> **Brown Corpus** — A Standard Corpus of Present-Day Edited American English,
> for use with Digital Computers.
> W. N. Francis and H. Kučera (1964), Department of Linguistics, Brown University,
> Providence, Rhode Island, USA. Revised 1971, Revised and Amplified 1979.
> <http://www.hit.uib.no/icame/brown/bcm.html>
>
> "Distributed with the permission of the copyright holder, redistribution
> permitted." (from the README distributed with the corpus via NLTK)

Obtained via NLTK (`nltk.download('brown')`, `nltk.corpus.brown.words()`).

## How these files were produced

Run [`../python_fedenanni/prepare_brown.py`](../python_fedenanni/prepare_brown.py).
It downloads the corpus, checks the size against the paper, tokenises it
(case preserved; 800k/200k/181,041 split; words with frequency ≤ 3 merged into
`<unk>`), and writes the files here. See `stats.json` for paper-vs-actual counts.

## Reference

Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin. *A Neural Probabilistic Language
Model.* Journal of Machine Learning Research 3:1137–1155, 2003.
(The paper itself is © JMLR and is not redistributed in this repository.)
