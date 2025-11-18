package TDA_Concretos;

import CL2.Lista1DLinkedL;
import C_Grafo.Connection;

public class ListaAristas extends Lista1DLinkedL {
    public boolean iguales(Object elementoL, Object elemento) {
        return false;
    }
    public String toString(){
        String str="Lista de Aristas: ";
        for(int i=0; i<tamanio();i++)
            str+=" nº"+i+" "+(Connection)devolver(i)+" ";
        return str;
    }
}
