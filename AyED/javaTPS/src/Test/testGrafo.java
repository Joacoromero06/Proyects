package Test; 
import C_Grafo.GrafoD;
import C_Grafo.GrafoND;

public class testGrafo {
    public static void main(String[] args) {
        GrafoND grafoND=new GrafoND("src/Test/ej1dijkstra.txt");
        GrafoD grafoD=new GrafoD("src/Test/ej1dijkstra.txt");
        //System.out.println(grafoND);
        //System.out.println(grafoND.Prim(0));
        //System.out.println(grafoND.Kruskal());
        System.out.println(grafoD);
        //grafoD.muestraDijkstra(3); 
        grafoD.muestraFloyd();grafoD.muestraCaminoFloyd(0, 4);
    }
}
