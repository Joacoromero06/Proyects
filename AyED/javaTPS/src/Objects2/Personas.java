package Objects2;

import java.util.Random;

import CL2.ConjuntoDLL;
import Objects1.Persona;

public class Personas extends ConjuntoDLL {

    public boolean iguales(Object elementoL, Object elemento) {
        return ((Persona)elementoL).iguales((Persona)elemento);
    }
    public void cargaPersonas(){
        int n =20+new Random().nextInt(4);
        for(int i=0;i<n;i++)
            meter(new Persona());
    }
    public String toString(){
        String str="Mostrando: \n";
        for(int i=0;i<tamanio();i++)
            str+=(Persona)devolver(i)+"\n";
        return str;
    }
    public Personas Clonar(){
        Personas clon =new Personas();
        for(int i=0;i<tamanio();i++)
            clon.meter(devolver(i));
        return clon;
    }
    
}
