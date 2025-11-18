package CL2;

import TDA.Conjunto;

public abstract class ConjuntoDLL extends Lista1DLinkedL implements Conjunto {
    
    public void meter(Object elemento){
        if(!pertenece(elemento)){
            insertar(elemento, tamanio());
        }
    }
    public boolean pertenece(Object elemento){
        return buscar(elemento)!=-1;
    }
    public void sacar(Object elemento){
        if(pertenece(elemento)){
            eliminar(buscar(elemento));
        }
    }
    public ConjuntoDLL Union(ConjuntoDLL conj2){
        ConjuntoDLL nuevo=Clonar();
        for(int i=0;i<conj2.tamanio();i++)
            nuevo.meter(conj2.devolver(i));
        return nuevo;
    }
    public ConjuntoDLL Interseccion(ConjuntoDLL conj2){
        ConjuntoDLL nuevo=Clonar();
        int i=0;
        while (i<nuevo.tamanio()) {
            if(!conj2.pertenece(nuevo.devolver(i)))
                nuevo.sacar(nuevo.devolver(i));
            else
                i++;
        }
        return nuevo;
    }
    public ConjuntoDLL Diferencia(ConjuntoDLL conj2){
        ConjuntoDLL nuevo=Clonar();
        for(int i=0;i<conj2.tamanio();i++)
            if(nuevo.pertenece(conj2.devolver(i)))
                nuevo.sacar(conj2.devolver(i));
        return nuevo;
    }
    public boolean Inclusion(ConjuntoDLL conj2){
        boolean b=true;
        int i=0;
        while (b&&i<tamanio()) {
            if(!conj2.pertenece(devolver(i)))
                b=false;
            i++;
        }
        return b;
    }
    public abstract ConjuntoDLL Clonar();
}
