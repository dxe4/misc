#lang racket
(define (my-map func lst)
(cond [(empty? lst) empty]
[else (cons (func (first lst))
            (my-map func (rest lst)))]))

(my-map add1 '(1 2 3 4))

((lambda (num) (- num 2)) 5)

(define (my-filter pred lst)
  (cond [(empty? lst) empty]
        [(pred (first lst))
         (cons (first lst) (my-filter pred (rest lst)))];book forgot to pass arg pred(or maybe i missed something?)
        [else (my-filter pred (rest lst))]));book forgot to pass arg pred(or maybe i missed something?)

(my-filter (lambda (i) (> i 5)) `(1 2 3 4 5 6 7 8))

;(define (my-filter2 pred lst)
;  (cond [(empty? lst) empty]
;        [(pred (first lst))
;         (cons (first lst) (my-filter2 (rest lst)))]
;        [else (my-filter2 (rest lst))]))
;
;(my-filter2 (lambda (i) (> i 5)) `(1 2 3 4 5 6 7 8))