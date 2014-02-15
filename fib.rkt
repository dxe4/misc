#lang racket
(define args (current-command-line-arguments))
(define (fib n)
  (define (fib-iter a b count)
    (if (= count 0)
        b
        (fib-iter (+ a b) a (- count 1))))
  (fib-iter 1 0 n))
(for ([i args])
  (display  
   (string-append  
    "\n" i " -> " (number->string (fib (string->number i)))))
  )