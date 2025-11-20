package Test;

import TDA_Concretos.*;

public class TestListaInt {
    public static void main(String[] args) {
        //LISTA COMUN
        /*ListaIntComun lis=new ListaIntComun();
        lis.CargaListaComun();
        System.out.println(lis);*/
        //LISTA ORDENADA
        ListaIntOrdenada lis2=new ListaIntOrdenada();
        lis2.CargaListaOrdenada();
        System.out.println(lis2);
    }   
}
