--RECURSION GUARDS PATTERNS--
CmpChar:: char -> char -> Bool
CmpChar c1 c2 = c1 == c2

palindromo:: String -> Bool
palindromo [] = True
palindromo [x] = True
palindromo (x:xs) = tail xs == x && palindromo 