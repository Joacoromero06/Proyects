for i in range(4):
    if(i%2==0):
        hola=[(1,{2,2,"3"}),2,2,"3"]
        print(hola)
    else:
        hola={("ala","atio"),2,"1",2}
        print(hola)
persona={
    1:"Joaquin",
    "apellido":"Romero",
    3:["juan","bauti"],
    (1,2):"Barrio Huaico",
    4:{
        "provincia":"Capital",
        "ciudad":"Salta"
    }
}
print(persona[(1,2)])
print(persona[3][0])

persona={}     
persona[1]="Salta"
persona[("cla","ve")]="huaico"
print(persona)
class Persona:
    def __init__(self,nombre,ubicacion):
        self.nombre=nombre
        self.ubicacion=ubicacion
    def presenta(self):
        print("Hola me llamo:"+self.nombre+" vivo en ",self.ubicacion) 
    def presenta2(self):
        print("Hola me llamo:"+self.nombre+" vivo en ",self.ubicacion[0]," en la ciudad ",self.ubicacion[1]) 
        
p=Persona("joaco",("Salta","Capital"))
p2=Persona("joaco",["Salta","Capital"])
p.presenta2()
p2.presenta()