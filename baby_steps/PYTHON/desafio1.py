#Dada una lista de números del 1 al 20, construí una nueva lista con los cuadrados solo de los múltiplos de 3.
l=[i for i in range(1,21)]
rtdo=[x*x for x in l if x%3==0]
print("lista de num 1 al 20: ",l)
print("lista con los cuadrados solo de los múltiplos de 3: ",rtdo)

import random
#Dada la lista [4, 7, 10, 3, 8], generá una nueva lista con la palabra "par" o "impar" según corresponda
l=[random.randint(1,10) for i in range(5)]
rtdo=["par" if x%2==0 else "impar" for x in l]
print("lista de num randoms del 1 al 10: ",l)
print("lista describiendo los numeros de l:",rtdo)

#Dada una matriz construí una lista plana que contenga solo los números pares mayores que 2.
longitudes=[random.randint(1,3) for i in range(3)]
m=[[random.randint(0,6) for j in range(i)] for i in longitudes]
l=[x for f in m for x in f if x>2 and x%2==0]
print("matriz irregular de 3 filas es: ",m)
print("lista plana que contenga solo los números pares mayores que 2: ",l)

#Dada una lista de palabras, construí una nueva lista donde: 
# si la palabra tiene más de 4 letras → se guarda al revés 
# si tiene 4 letras o menos → se guarda como está
nombres = ["Juan", "María", "Carlos", "Ana", "Luis", "Laura", "Pedro", "Sofía", "Javier", "Elena"]
l=[random.choice(nombres) for i in range(5)]
rtdo=[x[::-1] if len(x)>4 else x for x in l]
print("nombres aleatorios: ",l)
print("si el nombre tiene mas de 4 letras se invierte: ",rtdo) 

#Dada una lista con distintos tipos Construí una lista de strings con más de 3 letras.
 
