package Test;

import TADS_Adittion.AddRac;
import TADs.Numero;
import TADs.Racional;

public class TestRac {
    public static void main(String[] args) {
        Racional fracc1=new Racional(new Numero(20), new Numero(5));
        Racional fracc2=new Racional(new Numero(49), new Numero(14));
        fracc1.Simplificacion();System.out.println(fracc1);
        fracc2.Simplificacion();System.out.println(fracc2);

        Racional fraccaux=new Racional(new Numero(),new Numero());
        fraccaux.Producto(fracc2, fracc1);System.out.println(fraccaux);

        fraccaux=new Racional(new Numero(),new Numero());
        fraccaux.Sumar(fracc2, fracc1);System.out.println(fraccaux);

        fraccaux=new Racional(new Numero(),new Numero());
        fraccaux.Potencia(fracc2, 3);System.out.println(fraccaux);

        //Uso de AddRac
        AddRac fracc4 =new AddRac();
        fracc4.Division(fracc1, fracc2);System.out.println(fracc4);
    }
}
