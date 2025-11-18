package CL1;

import CL2.Lista2DLinkedL;
import TDA.OperacionesCL1;

public abstract class ColaPrioridadDLL extends Lista2DLinkedL implements OperacionesCL1{

    public void meter(Object elemento) {
        insertar(elemento);
    } 
    public Object sacar() {
        return devolver(0);
    }
    public boolean estaVacia(){
        return tamanio()==0;
    }
    public void limpiar(){
        limpiar();
    }
   
    public abstract boolean iguales(Object elemento1, Object elemento2);
	public abstract boolean esMenor(Object elemento1, Object elemento2);
	public abstract boolean esMayor(Object elemento1, Object elemento2);
    
}
