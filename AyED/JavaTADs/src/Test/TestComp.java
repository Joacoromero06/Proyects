package Test; 

import TADS_Adittion.AddComp;
import TADs.Complejo;

public class TestComp {
    public static void main(String[] args) {
        
        Complejo n1=new Complejo(1,1);
        System.out.println(n1);
        
        //Uso AddComp
        AddComp n2=new AddComp();
        n2.Division((Object)n1,(Object)new Complejo());
        
        System.out.println(Complejo.Modulo(n1));
        
        n2.Conjugado(n1);
        System.out.println(n2);
    }
}
