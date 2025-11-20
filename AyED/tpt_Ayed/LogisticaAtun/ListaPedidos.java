package LogisticaAtun;

import java.util.Random;

import CL2.Lista1DLinkedL;

public class ListaPedidos extends Lista1DLinkedL{
    private int cantNatural,cantAceite,cantTotal;

    public ListaPedidos(){
        super();
        this.cantNatural=this.cantAceite=this.cantTotal=0;
    }
    public void SimulaPedidos(int n){ //n: cantidad de posibles clientes (los que estan en el mapa de salta)
        for(int i=0;i<n;i++){ 
            if(i==n-1)
                insertar(new Pedido(i), tamanio());     
            else{
                if(new Random().nextBoolean())
                    insertar(new Pedido(i), tamanio());       
                
            }
        }
    } 
    public void insertar(Pedido p,int pos){
        super.insertar(p, pos);
        this.cantNatural+=p.getCantNatural();
        this.cantAceite+=p.getCantAceite();
        this.cantTotal+=p.getCantTotal();

    }
    public String toString(){
        String str="Pedidos del Dia.\nTotal de latas: "+this.cantTotal+"\nCantidad al natural: "+this.cantNatural+"\nCantidad al aceite: "+this.cantAceite+"\n";
        for (int i=0;i<tamanio();i++)
            str+= devolver(i)+"\n";
        return str;
    }
    public boolean iguales(Object elementoL, Object elemento) {return ((Pedido)elementoL).iguales((int)elemento);}
    public Pedido devolver(int pos){return ((Pedido)(super.devolver(pos)));}

    public void marcaEntregado(int cliente){
        int pos=buscar(cliente);
        devolver(pos).marcar();
    }
}
