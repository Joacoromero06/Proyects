package TDA_Concretos;

import CL2.Lista1DLinkedL;

public class ListaBoolean extends Lista1DLinkedL {
    public boolean iguales(Object elementoL,Object elemento){
        return (boolean)elementoL==(boolean)elemento;
    }
    public Boolean devolver(int pos){
        return ((Boolean)(super.devolver(pos)));
    }
    public String toString() {
       String str="Vamos a mostrar la lista: [ ";
        for(int i=0;i<tamanio()-1;i++){
            str+=(Boolean)devolver(i);
            str+=",";
        }
        if(!estaVacia())
            str+=(Boolean)devolver(tamanio()-1);
        str+=" ]\n";
        return str;
    }
}
