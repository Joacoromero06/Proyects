package CL2;
import TDA.*;

//IMPLEMENTACION DE LISTA ORDENADA DE OBJETOS CON DLL  
public abstract class Lista2DLinkedL extends Lista0DLinkedL implements OperacionesCL4 {
	
	// insercion ordenada
	public void insertar(Object elemento) {
		NodoDoble nodo;

		if (estaVacia()) {
			this.frenteL = this.finalL = new NodoDoble(elemento);
			
		} else{
			if (esMenor(elemento, this.frenteL.getNodoInfo())) {		//insercion al frente
				this.frenteL = new NodoDoble(elemento, null, this.frenteL); // nuevo frente
				this.frenteL.getNextNodo().setPrevNodo(this.frenteL); // vamos al 2do y conectamos con el 1ero
				
			}else{
				if (esMayor(elemento, this.finalL.getNodoInfo()) || iguales(elemento, this.finalL.getNodoInfo())) {	//insercion al final, si es igual no puede ponerse antes.
					this.finalL=new NodoDoble(elemento,this.finalL,null);
					this.finalL.getPrevNodo().setNextNodo(this.finalL);
				}else{		// al medio		
					NodoDoble temp = this.frenteL;
					boolean flag = false;
					while (temp.getNextNodo() != null && !flag) {
						if (esMayor(elemento, temp.getNextNodo().getNodoInfo()) || iguales(elemento, temp.getNextNodo().getNodoInfo())) {
							temp = temp.getNextNodo();
						}else{
							flag = true;
						}
					}				
					// insercion al medio. entre temp y temp.next				
					nodo = new NodoDoble(elemento, temp, temp.getNextNodo());
					temp.getNextNodo().setPrevNodo(nodo);
					temp.setNextNodo(nodo);	
				}
			}
		}
		
		this.ultimo++; // incrementamos "ultima posicion" de lista
	}	
	
	public abstract boolean iguales(Object elemento1, Object elemento2);
	public abstract boolean esMenor(Object elemento1, Object elemento2);
	public abstract boolean esMayor(Object elemento1, Object elemento2);
	
	// implementar alg�n metodo de b�squeda
	public int buscar(Object elemento){
		// implementar!!!
		return -1;
	}
	
}
