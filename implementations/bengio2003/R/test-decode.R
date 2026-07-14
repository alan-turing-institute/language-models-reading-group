

## Vocabulary is a vector of words, 1-indexed
vocab <- scan("data/vocab.txt", what = character(), quote = NULL)

## Text is a vector of integers
train <- scan("data/train.ids", what = integer()) + 1

## First 64 words
vocab[head(train, n = 64)]
