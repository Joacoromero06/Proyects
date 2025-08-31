import random

#CLASE 1
def saludar(nom,ape):
    print(f"Buendia {nom} mi {ape}")
def pide_nombre():
    return input("Ingrese su nombre: ")
def getResto(n):
    return n%2
def pide_numero():
    return int(input("Ingrese numero: ")) 

#CLASE 2
def num_del_1_10():
    r1=random.randint(1,11)
    return r1
def inicializa_lista():
    L= [num_del_1_10() for i in range(1,11)]
    return L
def muestra_lista_potencias(lista):
    for c in lista:
        print(c, c**2, c**3)
def LeeNotas():
    L=[int(input("Ingrese Nota: "))for i in range(1,6)]
    return L
def MuestraMenor(lista):
    menor=lista[0]
    for i in lista:
        if(i<menor):
            menor=i
    print(menor)
def MuestraMayor(lista):
    mayor=0
    for i in lista:
        if(i>mayor):
            mayor = i
    print(mayor)
def CargaCadena():
    L=[input("Ingrese una palabra: ")for i in range(1,6)]
    return L
def InvierteLista(lista):
    Inv=[lista[i] for i in range(len(lista)-1,-1,-1)]
    return Inv
 