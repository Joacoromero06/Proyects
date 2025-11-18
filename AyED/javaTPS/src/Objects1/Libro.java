package Objects1;

import java.util.Random;

public class Libro {
    private int peso, beneficio,nro;

    public Libro(int peso, int beneficio,int nro) {
        this.peso = peso;
        this.beneficio = beneficio;
        this.nro=nro;
    }
    public Libro(int nroLibro){
        this(1+new Random().nextInt(11), 1+ new Random().nextInt(11) ,nroLibro);
    }
    
    public int GetPeso(){
        return this.peso;
    }
    public int GetBen(){
        return this.beneficio;
    }
    
    public boolean iguales(Libro l2){
        return this.nro==l2.nro;
    }
    public String toString(){
        return "Libro: "+this.nro+" pesa "+this.peso+" beneficia "+this.beneficio;
    }
    public Libro Clonar(){
        return new Libro(this.peso,this.beneficio,this.nro);
    }
    
}
