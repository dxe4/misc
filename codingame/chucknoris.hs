import System.IO
import Control.Monad
import Data.Char (ord)
import Data.List (intercalate)

main :: IO ()

solve' :: Float ->Float->[Float]
solve' number pos
    | pos == 0.5 = []
    | number >= pos = 1:solve' (number-pos) (pos/2)
    | otherwise = 0:solve' number (pos/2)

make_key num
    | num == 1.0 = " 0 0"
    | otherwise = " 00 0"
    
last_num prev num
    | prev == num = "0"
    | otherwise = make_key num
    
cacl_out :: Float ->[Float]->[[Char]]
cacl_out prev (n:ns)
    | null(ns) ==True = [(last_num prev n)]
    | n /= prev = (make_key n):(cacl_out n ns)
    | otherwise = "0":(cacl_out n ns)
    
process_input (c:cn) previous
    |null(cn) == True = res2
    | otherwise = res2 ++ (process_input cn (last res))
    where
        num = fromIntegral (ord c) :: Float
        res = solve' num 64.0
        res2 = cacl_out previous res
 

main = do
    hSetBuffering stdout NoBuffering -- DO NOT REMOVE
    
    -- Auto-generated code below aims at helping you parse
    -- the standard input according to the problem statement.
    
    message <- getLine
    
    -- hPutStrLn stderr "Debug messages..."
    hPutStrLn stderr message
    let bytes = reverse [ 2**x | x <- [0..6]]
    --- let char = head message
    ---let num = fromIntegral (ord char) :: Float
    let result = process_input message 999

    let z = intercalate "" result
    putStrLn (tail z)
