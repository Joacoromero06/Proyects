package TDA_Concretos;

import CL2.*;
import assets.Entrada;

public class ListaIntComun extends Lista1DLinkedL {

    public void CargaListaComun(){
        System.out.println("Ingrese la cantidad de elementos de la lista");
        int n=Entrada.sc.nextInt(),aux;
        for(int i=0;i<n;i++){
            System.out.println("Ingrese un entero a la lista: ");
            aux=Entrada.sc.nextInt();
            System.out.println("Posicion: ");
            this.insertar(aux, Entrada.sc.nextInt());
        }
    }

    public String toString() {
       String str="Vamos a mostrar la lista: \n[ ";
        for(int i=0;i<tamanio()-1;i++){
            str+=(Integer)devolver(i);
            str+=",";
        }str+=(Integer)devolver(tamanio()-1);str+=" ]\n";
        return str;
    }
    
    public boolean iguales(Object elementoL, Object elemento) {
        return (Integer)elementoL==(Integer)elemento;
    }
    

}