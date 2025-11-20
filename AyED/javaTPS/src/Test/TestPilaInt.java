package Test;

import TDA_Concretos.*;

public class TestPilaInt {
    public static void main(String[] args) {
        
        PilaInt pila=new PilaInt();
        pila.Carga();
        System.out.println(pila);

        pila.Reemplaza(2, 4);
        System.out.println(pila);

        System.out.println(PilaInt.Verifica("(a(s(s)))()(sad))" ));

        ColaInt capicuas=pila.GeneraModifica();
        System.out.println(capicuas);System.out.println(pila);
    }
}
