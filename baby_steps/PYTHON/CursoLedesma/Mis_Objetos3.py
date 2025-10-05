import random
class Persona:
    def __init__(self,nombre,edad,ciudad,numeros):
        self.nombre=nombre
        self.edad=edad
        self.ciudad=ciudad
        self.numeros=numeros
    def presentarse(self):
        print(f"Hola soy {self.nombre} tengo {self.edad} vivo en {self.ciudad} los nros q me tocaron son: {self.numeros}")
    def EsMayor(self):
        return self.edad>18
    def CambiaCiudad(self):
        nvo=input("Ingrese nueva ciudad: ")
        self.ciudad=nvo      
def crea():
    nombre=input("Ingrese nombre: ")
    edad=int(input("Ingrese la edad: "))
    ciudad=input("Ingrese ciudad: ")
    numeros=[random.randint(0,100) for i in range(0,5)] 
    uno = Persona(nombre,edad,ciudad,numeros)
    return uno
def AgregaVector(nvo,vector:list):
    return vector.append(nvo)
def DetGanador(vector:list,persona):
 """usar all del github""" 