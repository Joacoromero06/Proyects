package Objects2;

import java.util.Random;

import CL2.ConjuntoDLL;
import Objects1.Libro;

public class MochilaLibros extends ConjuntoDLL{
    private int capacidad,beneficioTotal,capOrg;
    
    public MochilaLibros(int capacidad){
        limpiar();
        this.capacidad=capacidad;
        this.beneficioTotal=0;
        this.capOrg=this.capacidad;
    }
    public MochilaLibros(){
        this(1 + new Random().nextInt(31));
    } 

    public boolean iguales(Object elementoL, Object elemento) {
        return((Libro)elementoL).iguales((Libro)elemento);
    }
    
    public void meter(Libro l1){
        if(capacidad-l1.GetPeso()>=0){
            super.meter(l1);
            this.capacidad-=l1.GetPeso();
            this.beneficioTotal+=l1.GetBen();
        }
    } 
    public void sacar(Libro l1){
        if(pertenece(l1)){
            eliminar(buscar(l1));
            this.beneficioTotal-=l1.GetBen();
            this.capacidad+=l1.GetPeso();
        }
    }

    public int GetBeneficioTotal(){
        return this.beneficioTotal;
    }
    public int GetCapacidad(){
        return this.capacidad;
    }

    public MochilaLibros Clonar(){
        MochilaLibros clon=new MochilaLibros(this.capOrg);
        for(int i=0;i<tamanio();i++){
            clon.meter(((Libro)this.devolver(i)).Clonar());
        }
        return clon;
    }
    
    public boolean esMejor(MochilaLibros m2){
        return this.beneficioTotal>m2.beneficioTotal;
    }
    
    public String toString(){
        String str="Mochila:\n ";
        for(int i=0;i<tamanio();i++)
            str+=(Libro)devolver(i)+"\n";
        str+="Beneficio total: "+this.beneficioTotal+" Capacidad: "+this.capacidad;
        return str;
    }
}
