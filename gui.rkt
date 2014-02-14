#lang racket
(require 2htdp/universe 2htdp/image)

(big-bang 0
          (on-tick add-3-to-state)
          (to-draw draw-a-ufo-onto-an-empty-scene))

(define (add-3-to-state current-state)
  (+ current-state 3))

(define (draw-a-ufo-onto-an-empty-scene current-state)
  (place-image IMAGE-of-UFO (/ WIDTH 2) current-state
             (empty-scene WIDTH HEIGHT)))