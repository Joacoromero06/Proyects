package Test;

import Recursos.ParOrdenado;
import TADS_Adittion.AddNum;
import TADs.Numero;

public class TestNum {
    public static void main(String[] args) {
        Numero n1=new Numero(-15);Numero n2=new Numero(5);

        ParOrdenado cr= Numero.CyR(n1,n2);
        System.out.println("Cociente y resto entre  "+n1+" con "+n2+" es "+ cr);

        Numero n3=new Numero();
        n3.MCD(n1,n2);
        System.out.println("mcd entre  "+n1+" con "+n2+" es "+n3);

        n3=new Numero();
        n3.MCM(n1,n2);
        System.out.println("mcm entre  "+n1+" con "+n2+" es "+n3);

        //Uso de AddNum
        AddNum n4=new AddNum();
        n4.Division(n1, n2);System.out.println(n4);
    }
}
