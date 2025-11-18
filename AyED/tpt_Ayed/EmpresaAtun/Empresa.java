package EmpresaAtun;

import GestionAtun.BrazoGestion;
import GestionAtun.ColaLatas;
import LogisticaAtun.BrazoLogistica;
import LogisticaAtun.ListaPedidos;
import LogisticaAtun.MapaClientes;
import LogisticaAtun.Repartidor;

public class Empresa {
    private PilaLatas Stack1,Stack2;//1 al natural. 2 al aceite
    private ColaLatas cinta;
    private BrazoGestion brazoGestion;
    //
    private BrazoLogistica brazoLogistica;
    private ListaPedidos pedidosDelDia;
    private Repartidor repartidor;
    public MapaClientes mapa;


    public Empresa(){
        this.Stack1=new PilaLatas(30);this.Stack2=new PilaLatas(30);
        this.cinta=new ColaLatas(51);//0-49 latas
        this.brazoGestion=new BrazoGestion(Stack1, Stack2);
        //
        this.brazoLogistica=new BrazoLogistica(Stack1, Stack2);
        this.pedidosDelDia=null;
        this.repartidor=new Repartidor();
        this.mapa=new MapaClientes("Test/GrafoClientes.txt");
        
    }
    public void SimulacionSemana(){
        for(int dia=1;dia<=5;dia++){
            switch (dia) {
                case 1:
                    System.out.println("\nTRABAJO DEL DIA LUNES");
                break;
                case 2:
                    System.out.println("\nTRABAJO DEL DIA MARTES");
                break;
                case 3:
                    System.out.println("\nTRABAJO DEL DIA MIERCOLES");
                break;
                case 4:
                    System.out.println("\nTRABAJO DEL DIA JUEVES");
                break;
                case 5:
                    System.out.println("\nTRABAJO DEL DIA VIERNES");
                break;
                default:
                    break;
            }
            if(dia==1||dia==3){
                gestionTrabaja_LyM();
                logisticaTrabaja_LyM();
            }
            if(dia==2||dia==4){
                logisticaTrabaja_MyJ();
            }
            if(dia==5){
                System.out.println("Viernes se descansa");
            }    
        }
    }
    
    public void gestionTrabaja_LyM(){
        getCinta().OperariosDepositan();
        System.out.println(getCinta());
        getBrazoGestion().Procesar(this.cinta);
    }
    public void muestraSituacion_Gestion(){
        System.out.println("SITUACION BRAZO DE GESTION: ");
        getBrazoGestion().MuestraSituacionBrazo();
        System.out.println(getCinta());
    }
    
    public void logisticaTrabaja_MyJ(){
        getRepartidor().Regresar();
        getRepartidor().getBolson().limpiar();
    }
    public void logisticaTrabaja_LyM(){ 
        setPedidosDelDia(new ListaPedidos());
        getPedidosDelDia().SimulaPedidos(this.mapa.getOrden());
        getBrazoLogistica().setListaPedidos(this.pedidosDelDia); 
        preparaRepartidor();
        getRepartidor().Entregar();
    }
    public void preparaRepartidor(){
        System.out.println("PREPARANDO AL REPARTIDOR: RUTAS-LISTA DE PEDIDOS-COMPARTIMIENTO");
        getBrazoLogistica().imprimeLista(repartidor);//el brazo le da la lista de pedidos
        getMapa().CalculaRutaH_Ida(9);//el mapaa calcula las rutas 
        getRepartidor().setRutaIda(mapa.getRutaHamiltoniana());//la empresa le da las rutas que tiene que seguir
        getMapa().CalculaRutaH_Vuelta(9);
        getRepartidor().setRutaVuelta(mapa.getRutaHamiltoniana());
        getBrazoLogistica().Procesar(this.repartidor);//el brazo le carga el compartimiento
        System.out.println(getRepartidor());
    } 
    
    public PilaLatas getStack1() {return Stack1;}
    public PilaLatas getStack2() {return Stack2;}
    public ColaLatas getCinta() {return cinta;}
    public BrazoGestion getBrazoGestion() {return brazoGestion;}
    public BrazoLogistica getBrazoLogistica() {return brazoLogistica;}
    public ListaPedidos getPedidosDelDia() {return pedidosDelDia;}
    public Repartidor getRepartidor() {return repartidor;}
    public MapaClientes getMapa() {return mapa;}
    
    public void setStack1(PilaLatas stack1) {this.Stack1 = stack1;}
    public void setStack2(PilaLatas stack2) {this.Stack2 = stack2;}
    public void setCinta(ColaLatas cinta) {this.cinta = cinta;}
    public void setBrazoGestion(BrazoGestion brazoGestion) {this.brazoGestion = brazoGestion;}
    public void setBrazoLogistica(BrazoLogistica brazoLogistica) {this.brazoLogistica = brazoLogistica;}
    public void setPedidosDelDia(ListaPedidos pedidosDelDia) {this.pedidosDelDia = pedidosDelDia;}
    public void setRepartidor(Repartidor repartidor) {this.repartidor = repartidor;}
    public void setMapa(MapaClientes mapa) {this.mapa = mapa;}
}
