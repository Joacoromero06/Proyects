package LogisticaAtun;

import java.util.Random;

public class Pedido{
    private int cliente;
    private boolean entregado;
    private int cantAceite, cantNatural, cantTotal;

    public Pedido(int cliente){//simula cuanto pidio
        this.cantAceite=new Random().nextInt(5);
        this.cantNatural=new Random().nextInt(5);
        this.cantTotal=this.cantAceite+this.cantNatural;
        this.entregado=false;
        this.cliente=cliente;
    }
    public String toString(){
        String str="\nPedido del Cliente "+ this.cliente+":\n"+"Cantidad de latas TOTALES: "+this.cantTotal+"\nLatas de atun al NATURAL: "+this.cantNatural+"\nLatas de atun al ACEITE: "+this.cantAceite+"\nEstado del pedido: " ;
        if(this.entregado)
            str+="Entregado";
        else
            str+="No entregado";
        return str;
    }

    public boolean iguales(int cliente){return this.cliente==cliente;}
    public int getCliente(){return this.cliente;}
    public void marcar(){this.entregado=true;}
    public boolean esEntregado(){return this.entregado;}
    public int getCantAceite(){return this.cantAceite;}
    public int getCantNatural(){return this.cantNatural;}
    public int getCantTotal(){return this.cantTotal;}


}