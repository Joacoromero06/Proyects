package Test;

import C_Grafo.GrafoD;

public class testGrafoD {
    public static void main(String[] args) {
        GrafoD migrafo=new GrafoD("Test/GrafoClientes.txt");
        //System.out.println(migrafo);
        //migrafo.CalculaRutaHamiltoniana(0, 9, 2);    
        //System.out.println(migrafo.getRutaHamiltoniana());
        migrafo.usarTriangInf();
        System.out.println(migrafo);
        migrafo.CalculaRutaHamiltoniana(0, 9, 2);    
        System.out.println(migrafo.getRutaHamiltoniana());
    }
    
    
}
