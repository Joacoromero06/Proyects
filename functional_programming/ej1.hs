import Data.Text.IO (hPutStrLn)
import Prelude hiding (gcd)
main :: IO()
main = putStrLn "HOlat mundo"
gcd a b = if b == 0 then a else gcd b (a `mod` b)
