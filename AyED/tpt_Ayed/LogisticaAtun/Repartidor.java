package LogisticaAtun;

import java.util.Random;

import TDA_Concretos.ListaIntComun;

public class Repartidor {
    private String nombre;
    private int estado;
    private Compartimiento bolson;
    private ListaPedidos listaPedidos;
    private ListaIntComun rutaIda, rutaVuelta;
    
    public Repartidor(String nombre,int estado,Compartimiento bolson, ListaPedidos listaPedidos,ListaIntComun rutaIda,ListaIntComun rutaVuelta) {
        this.nombre = nombre;
        this.estado=estado;
        this.bolson = bolson;
        this.listaPedidos = listaPedidos;
        this.rutaIda=rutaIda;
        this.rutaVuelta=rutaVuelta;
    }
    public static String GenerarNombre(){
        String[] nombres={"Juan", "María", "Carlos", "Ana", "Luis", "Laura", "Pedro", "Sofía", "Javier", "Elena", 
"Miguel", "Isabel", "José", "Carmen", "David", "Lucía", "Daniel", "Paula", "Alejandro", "Claudia"};
        return nombres[new Random().nextInt(nombres.length)];
    }
    public Repartidor(){
        this(GenerarNombre(),0,new Compartimiento(),new ListaPedidos(),new ListaIntComun(),new ListaIntComun());
    }

    public String toString(){
        String str= "\nRepartidor: "+this.nombre;
        if(estado==1)
            str+=" Esta entregando";
        if(estado==-1)
            str+=" Esta volviendo";
        if(estado==0)    
            str+=" Esta esperando a trabajar";
        str+="\nEsta llevando: "+this.bolson;
        str+="\nSu lista de pedidos es: "+ this.listaPedidos;
        if(estado==1)
            str+="\nLa ruta que esta siguiendo es: "+ this.rutaIda;
        if(estado==-1)
            str+="\nLa ruta que esta siguiendo es: "+ this.rutaVuelta;     
        if(estado==0)
            str+="\nLa ruta de ida es: "+ this.rutaIda+"La ruta de vuelta es: "+ this.rutaVuelta;     
        return str;
    }
    public void Entregar(){
        this.estado=1;
        System.out.println("REPARTIDOR EMPEZO A TRABAJAR");
        for(int i=0;i<rutaIda.tamanio();i++){
            System.out.print("\nRepartidor: Llegue a "+ rutaIda.devolver(i));
            entrega((int)rutaIda.devolver(i));
        }
        
    }
    public void entrega(int cliente){
        int pos=listaPedidos.buscar(cliente);
        Pedido pedido;
        if(pos!=-1){
            pedido=listaPedidos.devolver(pos);
            System.out.println(" este local si realizo un pedido este dia"+pedido); 
            System.out.print("ENTREGANDO....");
            System.out.println("Pedido entregado al cliente "+ cliente);
            this.bolson.saca(pedido.getCantNatural(),pedido.getCantAceite());
            this.listaPedidos.marcaEntregado(cliente);
        }
        else{
            System.out.print(" este local no realizo ningun pedido este dia. No entrego nada aqui.");
        }
        //System.out.println(listaPedidos.devolver(pos));//Visualizar que ocurrio
    }
    
    public void Regresar(){
        this.estado=-1;
        System.out.println("\nREPARTIDOR ESTA VOLVIENDO A LA FABRICA\n");
        for(int i=0;i<rutaVuelta.tamanio();i++)
            System.out.println("Repartidor: Llegue a "+ rutaVuelta.devolver(i));
        this.estado=0;
    }
    
    public void setBolson(Compartimiento bolson) {this.bolson = bolson;}
    public void setListaPedidos(ListaPedidos listaPedidos) {this.listaPedidos = listaPedidos;}
    public void setRutaIda(ListaIntComun rutaIda){this.rutaIda=rutaIda;}
    public void setRutaVuelta(ListaIntComun rutaVuelta){this.rutaVuelta=rutaVuelta;}
    public Compartimiento getBolson() {return bolson;}
    public ListaPedidos getListaPedidos() {return listaPedidos;}

    

}
