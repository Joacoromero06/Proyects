package Objects2;

import CL2.Lista2DLinkedL;
import Objects1.Seleccionado;
import assets.Entrada;

public class ListaSeleccionados extends Lista2DLinkedL{

    public boolean iguales(Object elemento1, Object elemento2) {
        return ((Seleccionado)elemento1).esIgual((Seleccionado)elemento2);
    }

    public boolean esMenor(Object elemento1, Object elemento2) {
        return ((Seleccionado)elemento1).esMenor((Seleccionado)elemento2);
    }

    public boolean esMayor(Object elemento1, Object elemento2) {
        return ((Seleccionado)elemento1).esMayor((Seleccionado)elemento2);
    }
    
    public void CargaSelecciones(){
        System.out.println("Ingrese cantidad de selecciones");
        int n=Entrada.sc.nextInt();
        Seleccionado aux;
        for(int i=0;i<n;i++){
            aux=new Seleccionado();
            aux.Carga1();
            this.insertar(aux);
        }
    }
    public String toString(){
        String str="Lista de Seleccionados:\n ";
        int orden=0;
        for(int i =tamanio()-1;i>=0;i--){
            str+=((Seleccionado)this.devolver(i)+" Orden: "+ orden+"\n");
            orden++;
        }
        return str;
    }
}
