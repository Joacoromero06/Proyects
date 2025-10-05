class Persona:
    def __init__(self,nombre,edad,gen):
        self.nombre=nombre
        self.edad=edad
        self.gen=gen
    def __add__(self,otro):
        return self.edad+otro.edad
    def __str__(self):
        return f"nombre: {self.nombre} edad {self.edad}"
    def __lt__(self,otro):
        return self.edad<otro.edad
    def __eq__(self,otro):
        return  self.gen==otro.gen     
    def Cumpleaños(self):
        self.edad=self.edad+1

def creapersona():
        x=int(input("Ingrese la edad: "))            
        nom=input("Ingrese el nombre: ")
        if(x<25):
            gen=0
        elif(x<=40):
            gen=1
        elif(x<=56):
            gen=2
        else:
            gen =3
        p=Persona(nom,x,gen)    
        return p
def detGen(num):
    if (num==0):
        return "Gen Z"
    elif(num==1):
        return "Milenial"
    elif(num==2):
        return "Gen X"
    elif(num==3):
        return "Baby boomers"

L=[creapersona() for i in range(0,5)]
if(L[0]<L[4]):
    print(f"{L[0]} es menor que {L[4]}")
if(L[3]==L[2]):
    print("Ambos son de la misma generacion")
    detGen(L[3].gen)
else:
    print(f"Son de distintas generacion {L[3]} es {detGen(L[3])} {L[4]} es {detGen(L[4])}")
