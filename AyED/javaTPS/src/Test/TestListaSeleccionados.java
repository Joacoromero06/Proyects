package Test;

import Objects2.ListaSeleccionados;

public class TestListaSeleccionados {
    public static void main(String[] args) {
        ListaSeleccionados ranking=new ListaSeleccionados();
        ranking.CargaSelecciones();
        System.out.println(ranking);
    }

}
