#lang racket
(require rackunit)
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

;;
(define (my-length a-list)
  (if (empty? a-list)
      0
      (add1 (my-length (rest a-list)))))
;;

(my-length `(1 2 3))
;;
(and (odd? 5) (odd? 7) (odd? 9))
;;mutation
(define _even #f)
_even
(or (odd? 4) (set! _even #t))
_even
;;
   ;(when (and file-modified (ask-user-about-saving))
   ;  (save-file))
;;unless
  ;(define filename "my-first-program.rkt")
  ;(unless (ask-user-whether-to-keep-file filename)
  ;  (delete-file filename))
;;
(if (member 4 (list 3 4 1 5)) '4-is-in 'not-in)
(member 1 '(3 4 1 5));'(1 5)

(struct point (x y) #:transparent); if transparent not included #point will be printed instead of point 0 0
(point 2 2)

(define (distance-to-origin p)
  (sqrt (+ (sqr (point-x p)) (sqr (point-y p)))))

(distance-to-origin (point 3 4))

;;
(define (eq-first-items list1 list2)
  (eq? (first list1) (first list2)))
(eq-first-items (cons 1 empty) (cons 2 empty))
(eq-first-items (cons 3 empty) (cons 3 empty))

(check-equal? (add1 5) 6)
;

(define WIDTH 100)
(define HEIGHT 200)

(define X-CENTER (quotient WIDTH 2))
(define Y-CENTER (quotient HEIGHT 2))

Y-CENTER; 100

(unless (> HEIGHT 0)
  (error 'guess-my-number "HEIGHT may not be negative"))

;

(struct posn (x y))
(struct rectangle (width height))
(define (inside-of-rectangle? r p)
  (define x (posn-x p))
  (define y (posn-y p))
  (define width (rectangle-width r))
  (define height (rectangle-height r))
  (and (<= 0 x) (< x width) (<= 0 y) (< y height)))


(inside-of-rectangle? (rectangle 100 100) (posn 10 10))
(inside-of-rectangle? (rectangle 100 100) (posn 10 200))
