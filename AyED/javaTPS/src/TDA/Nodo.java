package TDA;
//Objeto clase nodo usado par implementar SLinkedList
public class Nodo {
	private Object nodoInfo;//Es generica permite guardar cualquier cosa al ser de tipo object
	private Nodo nextNodo;//Este atributo permite que un nodo contenga Literalmente otro nodo, estructura recursiva
	

	//Constructor por eso se llama del mismo nombre que la clase
	public Nodo(Object nodoInfo) {
		this(nodoInfo,null);
		//llama al constructor de la clase que estamos this(..,..) que tenga 2 parametros
	} 
	

	//Constructor de nodo. Crea/Instancia un nodo y le asigna a cada atributo la informacion que le llega de parametros
	public Nodo(Object nodoInfo, Nodo nextNodo) {
		this.nodoInfo = nodoInfo;
		this.nextNodo = nextNodo; 
	}

	
	//Es llamado por un nodo, un objeto de la misma clase. Setea la info que llega en parametros en el atributo
	public void setNodoInfo(Object nodoInfo) {
		this.nodoInfo = nodoInfo; 
	}


	//Setea cual va a ser el siguiente nodo, es decir cual nodo va a contener
	public void setNextNodo(Nodo nextNodo) {
		this.nextNodo = nextNodo; 
	}
	

	//Retotna de la info, recordemos que es un objeto lo que guarda, retorna un OBJETO, guarda un OBJETO
	public Object getNodoInfo() {
		return this.nodoInfo; 
	}
	

	//Retorna del nodo (que llama el metodo) lo que contiene en el siguiente nodo(otro nodo) 
	public Nodo getNextNodo() {
		return this.nextNodo; 
	}
	
	
}
