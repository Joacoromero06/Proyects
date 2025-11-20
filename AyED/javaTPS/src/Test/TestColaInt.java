package Test;

import TDA_Concretos.ColaInt;;

public class TestColaInt{
    public static void main(String[] args) {
        ColaInt cola=new ColaInt();
        
        cola.Carga();
        System.out.println(cola);
        
        System.out.println(cola.Cardinal());

        cola.Invertir();
        System.out.println(cola);

        ColaInt cola2=new ColaInt();
        cola2.Carga();
        System.out.println(cola2);

        cola.Concatenar(cola2);
        System.out.println(cola);

        System.out.println(cola2);//no se destruyen las colas
        

    }
}