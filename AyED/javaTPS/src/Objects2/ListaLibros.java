package Objects2;

import java.util.Random;

import CL2.Lista1DLinkedL;
import Objects1.Libro;

public class ListaLibros extends Lista1DLinkedL{
    
    public void GeneraLista(){
        int n= 4+new Random().nextInt(10);
        for(int i=0;i<n;i++){
            insertar(new Libro(i+1), tamanio());
        }
    }
    public boolean iguales(Object elementoL, Object elemento) {
        return((Libro)elementoL).iguales((Libro)elemento);
    }
    public String toString(){
        String str="Lista: \n";
        for(int i=0;i<tamanio();i++)
            str+=(Libro)devolver(i)+"\n";
        return str;
    }
}
