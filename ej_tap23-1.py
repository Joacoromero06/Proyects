import sys 
def resuelve(tarros_rojos, tarros_azules):
    """
    Asignar la distribucion de tarros optima
    Determinar la diferencia entre la que mas pintura tien y la que menos
    """
    #Distribuir tarros de forma optima
    tarros_azules.sort(reverse=True)
    tarros_rojos.sort()
    """print(f'azules: \n{tarros_azules}\nrojos:\n{tarros_rojos}')"""
    n=len(tarros_rojos)

    #Determinar la cant de pintura mayor y menor
    cant_pintura_mayor=0
    cant_pintura_menor=100000000000000000000000000000
    for i in range (n):
        suma=tarros_rojos[i]+tarros_azules[i]
        cant_pintura_mayor=max(cant_pintura_mayor,suma)
        cant_pintura_menor=min(cant_pintura_menor,suma)
        
    #Calcula la diferencia entre la que mas y menos pintura tiene
    dif=cant_pintura_mayor-cant_pintura_menor
    """print(f'menor:{cant_pintura_menor} mayor:{cant_pintura_mayor}')"""
    return dif
def main():
    data=sys.stdin.read().split()
    data=list(map(int,data))
    n=data[0]
    tarros_rojos=[]
    tarros_azules=[]
    for i in range(1,n+1):
        tarros_rojos.append(data[i])
        tarros_azules.append(data[i+n])
    
    #diferencia entre la que obtuvo mayor cant de pintura y la menor para la distribucion optima
    diferencia=resuelve(tarros_rojos,tarros_azules)
    sys.stdout.write(f'{diferencia}\n')

#main
if __name__ == '__main__':
    main()