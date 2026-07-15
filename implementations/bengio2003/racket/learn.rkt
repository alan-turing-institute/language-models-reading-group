#lang racket/base

(require math/array)


#|

Bengio 2003

nC : context length (in the paper, the context length is n - 1)
dV : vocabulary size
dE : dimension of the embedding space
dH : dimension of the "hidden layer"

Model parameters
----------------

Y = b + W Cs + U tanh(d + H Cs) 
probs = softmax(Y)

C : An array of shape #(dV dE)
for each word, a vector of length dE representing that word's embedding

Cs : An array of shape #(nC dE)
for each word in the context, that word's representation from C

W : An array of shape #(dV (* nC dE))  
b : A vector of length dV

H : An array of shape #(dH (* nC dE))
d : A vector of length dH
U : An array of shape #(dV dH)

|#


(define (probabilities C H d W U b words)
  

  )


