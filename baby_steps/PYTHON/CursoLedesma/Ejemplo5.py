import random
class Persona:
    def __init__(self,nombre,pres):
        self.nombre=nombre
        self.pres=pres
    def puede_viajar(self,costo):
        return self.pres>=costo
def creapersona():
    p=Persona(input("Ingrese nombre: "),random.randint(0,1000))
    return p
costoviaje=random.randint(300,600)
print(f"costo de viaje es: {costoviaje}")
L=[creapersona()for i in range(0,3)]
i=0
while((i<=2) & (L[i].puede_viajar(costoviaje))):
    i=i+1
if(i<2):
    print("No viajan")
else:
    print("A todos les alcanzan")
