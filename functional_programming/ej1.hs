import Data.Text.IO (hPutStrLn)
main :: IO()
main = putStrLn "HOlat mundo"
gcd a b = if b == 0 then a else gcd b (a b)