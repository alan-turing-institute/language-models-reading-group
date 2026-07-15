#lang racket/base

(require racket/port
         racket/string
         racket/list
         math/array
         math/matrix
         math/distributions
         (only-in racket/math tanh))

(module+ main

  (define vocabulary
    (list->vector
     (call-with-input-file "../data/vocab.txt" port->lines)))
  
  (define train-text
    (map string->number
         (string-split
          (call-with-input-file "../data/train.ids" port->string))))

  ;; Quick check that the training text has loaded correctly
  ;; (string-join
  ;;        (map
  ;;         (λ (n) (vector-ref vocabulary n))
  ;;         (take train-text 64)))

  (displayln (format "Loaded vocabulary. nV = ~a" (vector-length vocabulary)))
  (displayln (format "Loaded training data. Length: ~a" (length train-text)))

  #|

  Initialise model parameters

  Notation: <T> is a tensor
  
  |#

  (define nC 5)  ; Context length. (In Bengio 2003, context length it is n - 1) 
  (define dV (vector-length vocabulary))
  (define dE 10) ; Dimension of embedding space 
  (define dH 60) ; Dimension of "hidden" layer

  (display "Initialising tensors ...")

  (define <C> (make-init-tensor2 dV dE))
  (display " <C>")
  
  (define <W> (make-init-tensor2 dV (* nC dE)))
  (display " <W>")
  
  (define <H> (make-init-tensor2 dH (* nC dE)))
  (display " <H>")
  
  (define <U> (make-init-tensor2 dV dH))
  (display " <U>")
  
  (define <b> (make-init-tensor1 dV))
  (define <d> (make-init-tensor1 dH))
  (displayln " <b> <d>.")
  

  ;; Do one computation of the probabilities

  ;; <Cs> is e1 ⊕ e2 ⊕ ··· ⊕ e_nC, where ei is the embedding of word i 
  
  (define context (take train-text nC))
  (define <ctx> (array-slice-ref <C> (list context (::))))
  (define <Cs>
    (array-reshape <ctx> (vector (array-size <ctx>) 1)))

 (define ps (bengio-p <Cs> <H> <d> <W> <U> <b>))

  
  
  ) ; main



;; --------------------------------------------------------------------------------


#|

Bengio 2013

y = b + W Cs + U tanh(d + H Cs)

;; Tensors ("the weights") are implemented simply as arrays
;; Vectors ("the biases") are implemented as 1-dimensional arrays

|#


;; Computes the probabilities
(define (bengio-p <Cs> <H> <d> <W> <U> <b>)

  ;; "Single Static Assignment form"
  ;; Should make the backprop version easier to figure out
  (define <1> (matrix* <H> <Cs>))
  (define <2> (matrix+ <d> <1>))
  (define <3> (matrix-map tanh <2>))
  (define <4> (matrix* <U> <3>))
  (define <5> (matrix* <W> <Cs>))
  (define <y> (matrix+ <b> <5> <4>))  ; Should be shape #(dV 1) !

  (define <exp-y> (matrix-map exp <y>))
  (define sum-exp-y (array-all-sum <exp-y>))
  (define <p> (matrix-map (λ (e) (/ e sum-exp-y)) <exp-y>))

  <p>

  )






;; Tensors are initialised with numbers drawn from a normal distribution
;; with variance 2 / (rows + cols), aka "Xavier initialisation"
;;
;; See, e.g., Glorot and Bengio, "Understanding the difficulty of training
;; deep feedforward neural networks" (2010)

;; nr nc : number of rows and columns respectively
(define (make-init-tensor2 nr nc)
  (define d (normal-dist 0 (sqrt (/ 2 (+ nr nc)))))
  (define initvs (sample d (* nr nc)))
  (list->matrix nr nc initvs))

;; Vectors are initialised with 0.0
;; These are "column vectors", ie, matrices of shape (d 1)
;; d is a number
(define (make-init-tensor1 d)
  (make-matrix d 1 0.0))


