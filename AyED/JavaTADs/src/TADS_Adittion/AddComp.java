package TADS_Adittion;

import Recursos.IOpeAvanzadas;

import TADs.Complejo;

public class AddComp extends Complejo implements IOpeAvanzadas{
    public AddComp(){
        super();
    }
    public void Division(Object a,Object b){//No es uso del Tad entre a los atributos
        Complejo aux=new Complejo();
        if(!Iguales(aux, b)){
            this.real=((Complejo)a).getReal()/((Complejo)b).getReal();//no tendria sentido en este caso q sea private
            this.imaginaria=((Complejo)a).getImaginaria()/((Complejo)b).getImaginaria();//Esta protected 
        }
        else
            System.err.println("Division por 0+0i no definida");
    }
}
