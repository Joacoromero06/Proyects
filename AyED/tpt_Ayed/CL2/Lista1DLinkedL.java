package CL2;
import TDA.*;

// IMPLEMENTACION LISTA COMUN DE OBJETOS CON DLL
public abstract class Lista1DLinkedL extends Lista0DLinkedL implements OperacionesCL3 {
	
	public void insertar(Object elemento, int posicion) {
		NodoDoble nodo;
		if (posicion > tamanio() || posicion < 0) {
			System.out.println("Error insertar. Posicion inexistente ");
		}else{
			if (posicion == 0) { // insercion al comienzo
				if (!estaVacia()) {
					this.frenteL = new NodoDoble(elemento, null, this.frenteL);
					this.frenteL.getNextNodo().setPrevNodo(this.frenteL);
				}else{
					this.frenteL = this.finalL = new NodoDoble(elemento);					
				}
			}else{
				if (posicion == tamanio()) { // insercion al fin
					this.finalL = new NodoDoble(elemento, this.finalL, null); // nuevo nodo fin
					this.finalL.getPrevNodo().setNextNodo(this.finalL); // reconexion penultimo nodo al nuevo fin
				}else{
					// insercion al medio					
					NodoDoble prev, next;
					prev = this.frenteL;
					next = this.frenteL.getNextNodo();
					for (int counter = 1; counter < posicion; counter++) {
						prev = prev.getNextNodo();
						next = next.getNextNodo();						
					}
					
					nodo = new NodoDoble(elemento,prev,next);
					prev.setNextNodo(nodo); // actualizamos referencias
					next.setPrevNodo(nodo);					
				}
			}			
			this.ultimo++; // incrementamos "ultima posicion" de lista
		}		
	}

	public void reemplazar(Object elemento, int posicion) {		
		if (estaVacia()) {
			System.out.println("Error reemplazar. Lista vacia...");
		} else {
			if (posicion >= tamanio() || posicion < 0) {
				System.out.println("Error reemplazar. La posicion no existe..");
			}else{
				if (posicion == 0) {
					this.frenteL.setNodoInfo(elemento);
				}else{
					if (posicion == tamanio() - 1) {
						this.finalL.setNodoInfo(elemento);
					}else {
						NodoDoble temp;
						temp = this.frenteL;
						
						for (int counter = 0; counter < posicion; counter++) {						
							temp = temp.getNextNodo();		
						}				
						
						temp.setNodoInfo(elemento);
					}
				}				
			}						
		}		
	}
	
	public abstract boolean iguales(Object elementoL, Object elemento);
	
	public int buscar(Object elemento) {		
		int posicion = -1; int contador = 0;
		Object unElemento;
		NodoDoble temp;
		
		temp = this.frenteL;
		while (temp != null && posicion == -1) {
			unElemento = temp.getNodoInfo();
			if (iguales(unElemento,elemento)) {
				posicion = contador;
			}else{
				temp = temp.getNextNodo();
				contador++;
			}
		}				
		return posicion;
	}

}
