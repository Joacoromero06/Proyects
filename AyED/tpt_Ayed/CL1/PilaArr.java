package CL1;
import TDA.*;

//IMPLEMENTACION CL1->PILA CON ARRAY
public class PilaArr implements OperacionesCL1, OperacionesCL1Estaticos {
	private Object[] pila;//Array de objetos llamado pila
	private int cabeza;//Manejamos un indice cabeza para no hacer corrimiento
	private int tamPila;//Necesitamos el tamaño del array para declararlo, no de elementos sino del array
	

	//Constructor de PilaArr
	public PilaArr(int tamPila) {
		this.tamPila = tamPila;
		this.pila = new Object[this.tamPila];
		limpiar(); //Limpiar asigna el valor al indice cabeza cuando PilaArr esta vacia como lo manejamos
	}
	

	//La cabeza es el ind en el ultimo elemento del vector pero el primero en la pila 
	//Logramos esto porque sacamos el ultimo elemento del vector
	//El ultimo elemento del vector es el primero en la pila, todo manejado por el indice cabeza
	public void meter(Object elemento) {
		if (!estaLlena()){//No puede estar lleno el vector donde se maneja la pila
		  incrementaCabeza();//La idea es cargarlo en el ind cabeza
		  this.pila[this.cabeza] = elemento;//Piso la basura que haya
		} else {
			System.out.println("Error. Pila llena..");
		}
	}
	

	public Object sacar() {
		Object elemento = null;//el objeto que vamos a retornar
		if (!estaVacia()){
			elemento = this.pila[this.cabeza];//Sacamos el primero en la pila(ultimo en el array)//SIGUE AHI ES BASURA
			decrementaCabeza();//Ahora el primero es el anterior
		} else {//Si esta vacia retornamos nulo y mostramos un mensaje
			System.out.println("Error. Pila vacia..");
		}
		return elemento; 
	}


	public void limpiar() {
		//Si el array tiene elementos luego se van a sobreescribir
		this.cabeza = -1; //El indice -1 nos dice que la pila esta vacia no cero 
	}//-1 Porque la idea es que en cabeza esta el ultimo elemento ingresado osea el primero en la pila
	//Por eso cuando metemos aumentamos cabeza y metemos ahi, no de otra forma sino podemos sobrescribir
	//Si fuera 0 la pila no estaria vacia, significa que en el primero en la pila esta en la posicion 0 del vector
	

	public boolean estaLlena() {
		return (this.cabeza == this.tamPila - 1);//Si el primero en la pila y ultimo entrado esta en la ultima posicion
	}	//Es porque ya esta llena la pila, si meto uno mas me salgo del array seria como que me salgo de la cajita donde esta la pila
		

	public boolean estaVacia() {
		return (this.cabeza == -1);//Misma idea para limpiar nada mas que chequea si esta vacia
	}
	
	private void incrementaCabeza() {
		this.cabeza++; 
	}
		
	private void decrementaCabeza(){
		this.cabeza--;
	}	
	
}
