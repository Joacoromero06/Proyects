import sys
def chequea_ganador(tablero):
    """
    Chequea si hay un ganador en el tablero del TATETI
    """    
    #Las posibles posiciones ganadoras
    ganadoras=[(1,2,3),(4,5,6),(7,8,9)
               ,(1,4,7),(2,5,8),(3,6,9)
               ,(1,5,9),(3,5,7)]
    for a,b,c in ganadoras:
        if tablero[a]!='.'and tablero[a]==tablero[b]==tablero[c]:
            return tablero[a]
    return '.'
def det_casillas_posibles(restricciones,tablero):
    movimientos=[]
    for i in range (1,10):
        band=True
        for a,b in restricciones:
            if tablero[a]=='.' and b==i:
                band=False
        if band and tablero[i]=='.':
            movimientos.append(i)
    return movimientos

    
def minimax_tateti_restricciones(tablero,restricciones,esmax):
    """
    Algoritmo minimax para simular el juego optimo de tateti
    Con restricciones en los posibles movimientos
    X es max
    O es min
    """
    #Caso Base
    if chequea_ganador(tablero)=='O':
        return -1
    if chequea_ganador(tablero)=='X':
        return 1

    #Determina los movimientos posibles y Caso Base
    movimientos=det_casillas_posibles(restricciones,tablero)
    if not movimientos:
        return 0
    
    #Backtrack para busqueda de la mejor jugada
    if esmax:
        jugada_maximiza_max=-2
        for a in movimientos:
            tablero[a]='X'
            jugando_a =minimax_tateti_restricciones(tablero,restricciones,False)
            jugada_maximiza_max=max(jugada_maximiza_max,jugando_a)
            #backtrack
            tablero[a]='.'
        return jugada_maximiza_max
    else:
        jugada_minimiza_max=2
        for b in movimientos:
            tablero[b]='O'
            jugando_b=minimax_tateti_restricciones(tablero,restricciones,True)
            jugada_minimiza_max=min(jugada_minimiza_max,jugando_b)
            #Backtrack
            tablero[b]='.'
        return jugada_minimiza_max

def main():
    data= sys.stdin.read().split()
    data=list(map(int,data))
    #Obtener el n pero no lo uso
    n=data[0]
    
    #Armarse restricciones
    c=0
    restricciones=[]
    for x in data:
        if c==0:
            c+=1
            continue 
        if c%2==1:
            restriccion_a=x
        else:
            tupla_ab=(restriccion_a,x)
            restricciones.append(tupla_ab)
        c+=1
    #Armar el tablero
    tablero=['.' for i in range(10)]
    #Simular partida perfecta
    rtdo=minimax_tateti_restricciones(tablero,restricciones,True)
    if rtdo==1:
        sys.stdout.write('X\n')
    elif rtdo==0:
        sys.stdout.write('E\n')
    elif rtdo==-1:
        sys.stdout.write('O\n')
if __name__=='__main__':
    main()
