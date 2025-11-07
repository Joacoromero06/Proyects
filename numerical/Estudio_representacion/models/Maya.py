class Maya:
    def __init__(self, base: int, t: int, L: int, U: int):
        """
        Construye un objeto 'maya' con sus 4 elementos fundamentales.
        t: cantidad de digitos significativos.
        L, U: son generan el rango de exponentes validos en la maya
        """
        self.b = base
        self.t = t
        self.L = L
        self.U = U
    def __str__(self) -> str:
        s: str = '\n---- Maya ----\n'
        s += f'\nBase: {self.b}'
        s += f'\nCantidad de digitos significativos: {self.t}'
        s += f'\nLimite inferior, exponente: {self.L}'
        s += f'\nLimite superior, exponente: {self.U}'
        return s

if __name__ == '__main__':
    print(Maya(16, 52, 307, 308))