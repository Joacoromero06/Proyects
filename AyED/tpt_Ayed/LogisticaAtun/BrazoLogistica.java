package LogisticaAtun;

import EmpresaAtun.BrazoRobotico;
import EmpresaAtun.Lata;
import EmpresaAtun.PilaLatas; 

public class BrazoLogistica extends BrazoRobotico{  
    private boolean alternar;
    private Compartimiento bolson;
    private ListaPedidos pedidosDelDia;
    
    public BrazoLogistica(PilaLatas A, PilaLatas B) {
        super(A, B);
        this.alternar=false;
        this.bolson=null;
        this.pedidosDelDia=null;
    }
    
    public void Procesar(Object procesado) { 
        Encender();
        cargaRepartidor((Repartidor)procesado);
    } 
    public void cargaRepartidor(Repartidor repartidor){//vamos a cargar en el mismo compratimiento
        setBolson(repartidor.getBolson());
        cargaCompartimiento();
    }
    public void cargaCompartimiento(){
        Lata lata=null;
        while (!Pare()) { 
            if(alternar){
                lata=getPilaA().sacar();
                alternar=!alternar;
                if(getPilaA().estaVacia())
                    Detener(1);
            }    
            else{
                lata=getPilaB().sacar();
                alternar=!alternar;
                if(getPilaB().estaVacia())
                    Detener(-1);
            }    
            Procesar1Lata(lata);
        }
    }
    public void Procesar1Lata(Lata lata) {//agrega balanceado 
        if(!getBolson().estaLleno()){
            if(lata.getEsNatural())
                getBolson().insertar(lata, 0); 
            else{
                getBolson().insertar(lata, bolson.tamanio());}
        }
        else
            Detener(0);
    } 

    public void Detener(int razon) { 
        System.err.println("Brazo de Logistica Detenido");
        if(razon==0)
            Aviso(0);
        if(razon==1)
            Aviso(1);
        if(razon==-1)
            Aviso(-1);
        Parar();
    } 
    public void Aviso(int razon) { 
        if(razon==0)
            System.err.println("Bolson del repartidor cargado");
        if(razon==1)
            System.err.println("Stack1 Vacio, no quedan latas al natural");    
        if(razon==-1)
            System.err.println("Stack2 Vacio, no quedan latas al aceite");
    }
    
    
    public void imprimeLista(Repartidor repartidor){//no tiene lista se la doy
        repartidor.setListaPedidos(pedidosDelDia);
    } 

    public void setListaPedidos(ListaPedidos pedidosDelDia){this.pedidosDelDia=pedidosDelDia;}
    public ListaPedidos getListaPedidos(){return this.pedidosDelDia;}
    public void setBolson(Compartimiento bolson){this.bolson=bolson;}
    public Compartimiento getBolson(){return this.bolson;}
}