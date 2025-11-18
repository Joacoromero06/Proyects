package LogisticaAtun;

import CL2.Lista1DLinkedL;
import EmpresaAtun.Lata;

public class Compartimiento extends Lista1DLinkedL{
    private int cantNatural,cantAceite,cantTotal;

    public Compartimiento(){
        super();
        this.cantAceite=this.cantNatural=this.cantTotal=0;
    }

    public void limpiar(){
        super.limpiar();
        this.cantAceite=this.cantNatural=this.cantTotal=0;
    }
    public void insertar(Lata lata,int pos){
        super.insertar(lata, pos);
        if(lata.getEsNatural())
            cantNatural++;
        else
            cantAceite++;
        cantTotal++;
    }
    public void saca(int cantSacarNatural,int cantSacarAceite){
        sacaBalanceado(cantSacarNatural, cantSacarAceite, true);
    }
    public void sacaBalanceado(int cantSacarNatural,int cantSacarAceite,boolean turnoNatural){
        if(cantSacarNatural==0&&cantSacarAceite==0){return;}
        if(cantSacarNatural>0&&turnoNatural){
            saca1Natural();
            sacaBalanceado(cantSacarNatural-1, cantSacarAceite, false);
        }
        else if(cantSacarAceite>0&&!turnoNatural){
            saca1Aceite();
            sacaBalanceado(cantSacarNatural, cantSacarAceite-1, true);
        }
        else{
            sacaBalanceado(cantSacarNatural, cantSacarAceite, !turnoNatural);
        }
            
    }
    public void saca1Natural(){
        int pos=buscar(true);
        if (pos!=-1){
           eliminar(pos);
           this.cantNatural--;
           this.cantTotal--;
        }    
        else
            System.err.println("No hay suficientes latas al natural para entregar");

    }
    public void saca1Aceite(){
        int pos=buscar(false);
        if (pos!=-1){
           eliminar(pos);
           this.cantAceite--;
           this.cantTotal--;
        }    
        else
            System.err.println("No hay suficientes latas al aceite para entregar");
    }
    public String toString(){
        String str="Bolson: ";
        if(estaVacia())
            str+="VACIO";
        else
            str+="\nCantidad de latas TOTALES: "+this.cantTotal+"\nCantidad de latas al NATURAL: "+this.cantNatural+"\nCantidad de latas al ACEITE: "+this.cantAceite;
        return str;
    }
    public boolean estaLleno(){return tamanio()==40;}
    public boolean iguales(Object elementoL, Object elemento) {return ((Lata)elementoL).mismoTipo((boolean)elemento);}
    public Lata devolver(int pos){return ((Lata)super.devolver(pos));}    
}
