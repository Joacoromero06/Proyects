package Recursos;

import TADs.Numero;

public class ParOrdenado {
    private Numero x,  y;


    public ParOrdenado(){
        this.x=new Numero();
        this.y=new Numero();
    }

    public ParOrdenado(TADs.Numero x, TADs.Numero y) {//No hace falta especificar el pakete de la clase
        this.x = x;
        this.y = y;
    }

    public Numero getX() {
        return x;
    }

    public void setX(Numero x) {
        this.x = x;
    }

    public Numero getY() {
        return y;
    }

    public void setY(Numero y) {
        this.y = y;
    }
    
    public String toString(){
        return "X: "+this.x+"  Y: "+this.y;
    }
}
