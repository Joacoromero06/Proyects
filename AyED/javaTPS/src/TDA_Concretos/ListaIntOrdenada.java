package TDA_Concretos;

import CL2.*;
import assets.Entrada;
public class ListaIntOrdenada extends Lista3SLLinkedL{

    public void CargaListaOrdenada(){
        System.out.println("Ingrese la cantidad de elementos de la lista");
        int n=Entrada.sc.nextInt();
        for(int i=0;i<n;i++){
            System.out.println("Ingrese un entero: ");
            insertar(Entrada.sc.nextInt());
        }
    }

    public String toString() {
       String str="Vamos a mostrar la lista: \n[ ";
        for(int i=0;i<tamanio()-1;i++){
            str+=(Integer)devolver(i);System.out.println("paso"+i);
            str+=",";
        }str+=(Integer)devolver(tamanio()-1);str+=" ]\n";
        return str;
    }
    
    public boolean iguales(Object elemento1, Object elemento2) {
        return (Integer)elemento1==(Integer)elemento2;
    }

    public boolean esMenor(Object elemento1, Object elemento2) {
        return (Integer)elemento1<(Integer)elemento2;
    }

    public boolean esMayor(Object elemento1, Object elemento2) {
        return (Integer)elemento1>(Integer)elemento2;
    }
}
