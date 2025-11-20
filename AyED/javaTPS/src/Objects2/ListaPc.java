package Objects2;

import CL2.Lista1DLinkedL;
import Objects1.Alumno;
import Objects1.Pc;

public class ListaPc extends Lista1DLinkedL{
    public void InicilizaLab(){
        for(int i=0;i<15;i++)
            this.insertar(new Pc(i+1), i);
    }
    public int BuscaPcLibre(){
        int b=this.buscar(false);
        if(b==-1)
            System.out.println("No hay Pc libre.");
        else
            System.out.println( "Hay Pc libre la nro: "+b);
        return b;
    }

    public boolean iguales(Object elementoL, Object elemento) {
        return ((Pc)elementoL).estadoPc((boolean)(elemento));
    }
    public void OcuparPc(int pos,Alumno alumno,int tiempoLlegada){
        ((Pc)(devolver(pos))).usar(alumno,tiempoLlegada);
    }
    public void PasaMin(){
        for(int i=0;i<tamanio();i++)
            ((Pc)devolver(i)).PasaMin();
    }
}
