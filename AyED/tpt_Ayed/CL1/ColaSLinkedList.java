package CL1;
import TDA.*;

//IMPLEMENTACION DE CL1->COLA CON SLL
public class ColaSLinkedList implements OperacionesCL1 {
	protected Nodo frenteC, finalC;//Necesito dos nodos para poder hacer referencia al nodo ultimo y al primero
	//Guardan los nodos que estan al final y al principio
	

	public ColaSLinkedList() {
		limpiar();
	}


	public boolean estaVacia() {
		return (this.frenteC == null);//Solo por frente porque cuando saco solo queda frente en NULL y 
	}//final tiene ese primer ultimo elemento en la cola por eso lo asigno a nulo despues de sacar y que quede vacia



	public void meter(Object elemento) {
		if (!estaVacia()) {
			this.finalC.setNextNodo(new Nodo(elemento));
			this.finalC = this.finalC.getNextNodo();
		}else{//Si esta vacia la cola el final y el frente tienen el mismo nuevo primer elemento
			this.frenteC= new Nodo(elemento);
			this.finalC=this.frenteC;
		}
	}
	

	public Object sacar() {
		Object elemento = null;
		if (!estaVacia()) {
			elemento = this.frenteC.getNodoInfo();
			this.frenteC = this.frenteC.getNextNodo();//Saco el primer elemento de la cola de enfrente ahora el frente es el sig
			if (estaVacia()) {
				this.finalC = null;//Final quedo con el primer elemento de la cola como ya salio, cola vacia 
			}
		}else{
			System.out.println("Error sacar. Cola vacia");
		}
		return elemento;
	}
	
	public void limpiar() {//Si ambos nodos estan en null significan que la cola esta vacia y todos los nodos se eliminan
		this.frenteC = this.finalC = null;//Si no hago que final este en NULL en Final tendria un nodo en el aire que ya elimine
	}	
	
}
