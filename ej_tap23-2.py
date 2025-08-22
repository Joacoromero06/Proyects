import sys
def resuelve(problemas,cant_necesarias):
    """
    Determina la cantidad de competencias posibles de armar con:
    p_i problemas para dificultad i
    c_i cantidad de problemas de dificultad i necesarias para armar una comppetencia
    """
    
    #Cantidad de competencias
    c=0
    #Determinar si puedo armar una competencia 1ra vez
    b=armar_competencia(problemas,cant_necesarias)

    #Armar competencias y contar
    while b != False:
        c+=1
        b=armar_competencia(problemas,cant_necesarias)
    return c

def armar_competencia(problemas,cant_necesarias):
    k=len(problemas)
    b=True
    for i in range(k):
        p_i=problemas[i]
        c_i=cant_necesarias[i]
        #Si NO me alcanza-> busco problemas mas faciles
        if p_i<c_i:
            pedido=pedir_problemas_faciles(problemas,i)

            #Si AUN asi no alcanza-> no es posible armar otra competencia
            if pedido+p_i<c_i:
                b=False
            
            #Si SI me alcanza-> actualizo usando optimamente problemas mas faciles que i
            else: 
                actualiza(problemas,c_i-p_i,i)

        #Si SI me alcanza-> los uso
        else:
            problemas[i]-=cant_necesarias[i]
    return b


def pedir_problemas_faciles(problemas,i):
    #problemas_hasta_i = problemas[0:i:1]
    cant_problemas=0
    for j in range(i):
        cant_problemas+=problemas[j]
    return cant_problemas

def actualiza(problemas,falta,i):
    j=i-1 
    while j>=0 and falta>0:
        aux=falta
        falta-=problemas[j]
        if falta >0:
            problemas[j]=0
        else:
            problemas[j]-=aux
        j-=1

def main():
    data=sys.stdin.read().split()
    data=list(map(int,data))
    k=data[0]
    problemas=[]
    cant_necesarias=[]
    for i in range(1,k+1,1):
        cant_necesarias.append(data[i])
        problemas.append(data[i+k])
    cant_competencias=resuelve(problemas,cant_necesarias)
    sys.stdout.write(f'{cant_competencias}\n')

if __name__ =='__main__':
    main()
        

