package TDA_Concretos;

import CL2.Lista1DLinkedL;

public class ListaDoubleComun extends Lista1DLinkedL{
    public boolean iguales(Object elementoL,Object elemento){
        return (double)elementoL==(double)elemento;
    }
    public String toString() {
    String str = "Vamos a mostrar la lista: [ ";
    for (int i = 0; i < tamanio(); i++) {
        str += devolver(i).toString();
        if (i < tamanio() - 1) {
            str += ", ";
        }
    }
    str += " ]\n";
    return str;
    }

}
