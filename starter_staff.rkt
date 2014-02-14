#lang racket
(zero? 1);check if on is zero -> #f
(symbol=? 'foo 'FoO); check if two symbols are the same
(expt 2 3); 2^3
(cons 1
      (cons 2 (cons 3 empty)));racket list implementation
(cons 1 2);cons '(1 . 2)
(define cell (cons 'a 'b))
(car cell);left part 
(cdr cell); right part

;same
(define animals (cons 'pork (cons 'beef (cons 'chicken '()))))
(list 'pork 'beef 'chicken)

(first animals);'pork
(rest animals);'(beef chicken)

(list 'cat (list 'duck 'bat) 'ant); nested list
(second animals)
;47

(struct student (name id# dorm)) ; define structure
(define foo (student 'Joe 1234 'NewHall)) ; create structure
(student-name foo) ; get student name

;;list of structs
(define mimi (student 'Mimi 1234 'NewHall))
(define nicole (student 'Nicole 5678 'NewHall))
(define rose (student 'Rose 8765 'NewHall))
(define eric (student 'Eric 4321 'NewHall))
(define in-class (list mimi nicole rose eric))
(student-id# (third in-class))
;;
(struct student-body (freshmen sophomores juniors seniors))
(define all-students
  (student-body (list foo (student 'Mary 0101 'OldHall))
                (list (student 'Jeff 5678 'OldHall))
                (list (student 'Bob 4321 'Apartment))
                empty))

(student-name (first (student-body-freshmen all-students)))
;view structure
(struct example2 (p q r) #:transparent)
(define ex2 (example2 9 8 7))
ex2
(string=? "hello world" "good bye");t
;;;; Summary
;• zero? checks whether its argument is 0
;• symbol=? comares symbols
;• student? check if instance
;number,string,image,boolean,list,cons,empty
;;;
(define (stupid-function x)
  (cons x '(1 2 3)))
(stupid-function `(1,2,3))
;
(if (= (+ 1 2) 3)
    'yup
    'nope);yup
(if (odd? 5) 'odd-number 'even-number); odd number


;nested if
(define x 7)
(if (even? x)	
    'even-number	
    (if (= x 7)
        5
        'odd-number))

;;cond!!
(cond [(= x 7) 5]
      [(odd? x) 'odd-number]
      [else 'even-number])