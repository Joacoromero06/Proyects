package GestionAtun;

import CL1.ColaArr;
import EmpresaAtun.Lata;

public class ColaLatas extends ColaArr{
    private int tamanio;

    public ColaLatas(int tam) {
        super(tam);
        this.tamanio=tam;
    }
    
    public void OperariosDepositan(){
        for(int i=1;i<this.tamanio;i++)
            Agrega1Lata();
    }
    public String toString(){
        ColaLatas aux=new ColaLatas(this.tamanio);
        String str="Mostrando Cinta de Latas:\n";
        Lata lata=null;
        if(!estaVacia()){
            while (!estaVacia()) {
                lata=sacar();
                str+=lata+"\n";
                aux.meter(lata);
            }
            while (!aux.estaVacia()) 
                meter(aux.sacar());
        }
        else
            str+= "VACIA";
        return str;
        
    }
    public void Agrega1Lata() {meter(new Lata());}
    public Lata sacar(){return ((Lata)super.sacar());}
    
}
