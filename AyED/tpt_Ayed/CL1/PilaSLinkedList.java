//Clase del pakete contenedores, para que sea un proyecto bien diseñadp
package CL1;
import TDA.*;

//IMPLEMENTACION CL1->PILA CON SLL
public class PilaSLinkedList implements OperacionesCL1 {
	//Para manejar una pila con SlinkedList solo necesitamos un nodo que es la CABEZA que apartir de ella se enlaza todo
	private Nodo pila;
	

	//Constructor que llama al metodo Limpiar
	public PilaSLinkedList() {
		//Cuando hace new se crea el objeto con el constructor de java luego pasa al control de este constructor
		//Este llama al metodo Limpiar, solo limpiar() porque es implicito que lo llama el objeto recien creado 
		limpiar(); //=this.limpiar()
	}
	

	//Crea un nodo con el elemento que quiero meter a la pila
	//Come es el nuevo primero ahora "pila" es el 2do
	//el nuevo nodo apunta al 2do elemento y ahora el es el primero
	public void meter(Object elemento) {
		this.pila = new Nodo(elemento, this.pila);
	} 


	//SLinkedList esta vacia solo si el primer nodo es NULL (notiene ninguna info)
	public boolean estaVacia() {
		return (this.pila == null);
	} 
	

	//Vacia la pila el garbage collector de java elimina los nodos que pueden haber quedado enlazados(nadie apunta al primero)
	public void limpiar() {
		this.pila = null;
	}


	//Retorna un Objeto con la info del primero
	public Object sacar() {
		Object elemento = null;//Inicializamos por si la pila esta vacia no hay 2 return
		if (!estaVacia()){//Me aseguro que hay nodo para despues obtener su info y sacar el primero
			elemento = this.pila.getNodoInfo();
			this.pila = this.pila.getNextNodo();//Asignamos a la var pila que tiene el primer elemento el nuevo primero
		}else{
			System.out.println("Error sacar. Pila vacia");}
		return elemento; 
	}
	

	//Un while que printea cada info de cada nodo
	public void muestra(){
		Nodo temp;//temp navega la SLinkedList
		if (!estaVacia()){//Recien despues saber que no esta vacia puedo asignar el trimero a temp
			temp = this.pila;
			while (temp != null){//Redefinir el metodo toString de java para cada objeto
				System.out.println("Elemento " + temp.getNodoInfo().toString());//toString esta implicito Java lo detecta
				temp = temp.getNextNodo();
			}
		}else{
			System.out.println("Error muestra. Pila vacia");
		}
	}
	
}