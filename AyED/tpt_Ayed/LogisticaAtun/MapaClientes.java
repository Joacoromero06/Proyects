package LogisticaAtun;

import C_Grafo.GrafoD;  
public class MapaClientes extends GrafoD {
    private int nodoFabrica; 

    public MapaClientes(String archivo) {
        super(archivo);
        this.nodoFabrica=0; 
    }
    public void CalculaRutaH_Ida(int destino){  
        usarTriangSup();  
        CalculaRutaHamiltoniana(this.nodoFabrica, destino, 2);
        usarTodaMatriz();

    }
    public void CalculaRutaH_Vuelta(int origen){
        usarTriangInf();
        CalculaRutaDijkstra(origen, this.nodoFabrica);
        //CalculaRutaHamiltoniana(origen, nodoFabrica, 2);
        usarTodaMatriz();
    }
//Metodos para calcular Ruta que pase por una coleccion de nodos, no todos.
    

    
}
