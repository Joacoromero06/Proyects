package TDA;

import C_Arbol.NodoArbolBinario;

public interface operacionesArbolB {
	public void meter(Object nodoInfo);
	public boolean esta(Object nodoInfo);
	public void sacar(Object nodoInfo);
	public boolean estaVacio();
	public void limpiar();
	public NodoArbolBinario devuelve(Object nodoInfo);
	public void muestraInOrder();
	public void muestraPreOrder();
	public void muestraPostOrder();
	public int cantHojas(NodoArbolBinario nodo);
	public int alturaArbol(NodoArbolBinario nodo);
	//faltan operaciones...
}


