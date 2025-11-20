// implementa las operaciones insercion y eliminacion de nodos sobre un ABB. Sigue siendo abstracto

package C_Arbol; 

public abstract class ABB extends ArbolBinario{
		
	public abstract boolean iguales(Object nodoA, Object nodoB);
	public abstract boolean menor(Object nodoA, Object nodoB);
	public abstract boolean mayor(Object nodoA, Object nodoB);
	
	public void meter(Object nodoInfo){
		NodoArbolBinario temp, anterior;
		NodoArbolBinario nodo = new NodoArbolBinario(nodoInfo);
		
		if (estaVacio()){
			this.root=nodo;
		}else{
			temp=this.root;
			anterior=null;
			while (temp!=null){
				anterior=temp;
				if (mayor(temp.getNodoInfo(),nodoInfo)){
					temp=temp.getLeftChild();
				}else{
					if(menor(temp.getNodoInfo(),nodoInfo)) {
						temp=temp.getRightChild();
					}else {//son iguales
						temp=null;
					}
				}
			}
			if(!iguales(anterior.getNodoInfo(),nodoInfo)) {
				if (menor(anterior.getNodoInfo(),nodoInfo)){
					anterior.setRightChild(nodo);
				}else{
					anterior.setLeftChild(nodo);
				}
			}						
		}		
	}
	 
	public NodoArbolBinario devuelve(Object nodoInfo){//corregir return si no esta MAL
		NodoArbolBinario nodo=null;
		
		if (!estaVacio()){
			nodo=devuelveNodo(nodoInfo);			
			if (nodo==null){
				System.out.println("Error devuelve. No esta...");
			}			
		}else{
			System.out.println("Error devuelve. arbol vacio...");
		}		
		return nodo;
	}
	private NodoArbolBinario devuelveNodo(Object nodoInfo){
		NodoArbolBinario nodo=null;
		boolean response=false;
		
		nodo=this.root;
		while (nodo!=null && !response){
			if (iguales(nodo.getNodoInfo(),nodoInfo)){
				response=true;
			}else {
				if (mayor(nodo.getNodoInfo(),nodoInfo)){
					nodo=nodo.getLeftChild();
				}else {
					nodo=nodo.getRightChild();
				}
			}
		}	
		return nodo;
	}
	
 	public boolean esta(Object nodoInfo){
		NodoArbolBinario nodo;
		boolean response=false;
		if (!estaVacio()){
			nodo=devuelveNodo(nodoInfo);
			if (nodo!=null){
				response=true;
			}			
		}else{
			System.out.println("Error esta. arbol vacio...");
		}		
		return response;
	}
	
	public void sacar(Object nodoInfo){
		NodoArbolBinario temp,anterior, auxiliar;
		boolean response=false;
		
		if (!estaVacio()){
			if (esta(nodoInfo)){
				anterior=null;
				temp=this.root;
				
				while (temp!=null && !response){					
					if (iguales(temp.getNodoInfo(),nodoInfo)){
						response=true;
					}else {
						anterior=temp;
						if (mayor(temp.getNodoInfo(),nodoInfo)){
							temp=temp.getLeftChild();
						}else {
							temp=temp.getRightChild();
						}
					}					
				}
				
				if (temp.getLeftChild()==null && temp.getRightChild()==null){//hoja
					if (anterior==null){
						limpiar();
					}else{
						if (anterior.getLeftChild()==temp){
							anterior.setLeftChild(null);
						}else{
							anterior.setRightChild(null);
						}
					}
				}else{
					if (temp.getLeftChild()!=null && temp.getRightChild()!=null){//dos hijos
						anterior=temp;
						auxiliar=temp.getLeftChild();
						while (auxiliar.getRightChild()!=null){// busco el mayor de los menores
							anterior=auxiliar;
							auxiliar=auxiliar.getRightChild();
						}
						temp.setNodoInfo(auxiliar.getNodoInfo());// el nodo se reemplaza por el mayor de los menores
						
						if (anterior==temp){
							anterior.setLeftChild(auxiliar.getLeftChild());
							// el nuevo hijo izq del nodo eliminado es su nieto izq
						}else{
							// el nuevo hijo derecho del padre del nodo "eliminado" es su nieto izq.
							anterior.setRightChild(auxiliar.getLeftChild());
						}				
						
					}else{
						if (temp.getRightChild()!=null){// hijo der
							if (anterior==null){
								// nodo raiz con hijo der
								this.root=temp.getRightChild();
							}else{
								if (anterior.getRightChild()==temp){
									anterior.setRightChild(temp.getRightChild());
								}else{
									anterior.setLeftChild(temp.getRightChild());
								}
							}
						}else{//hijo izq
							if (anterior==null){
								// nodo raiz con hijo izq
								this.root=temp.getLeftChild();
							}else{
								if (anterior.getRightChild()==temp){
									anterior.setRightChild(temp.getLeftChild());
								}else{
									anterior.setLeftChild(temp.getLeftChild());
								}
							}
						}
					}
				}				
	
			}else{
				System.out.println("Error sacar. No esta elemento..");
			}
		}else{
			System.out.println("Error sacar. arbol vacio...");
		}
		
	}
	public void meter(Object nodoInfo){
		meterRec(this.root,nodoInfo);
	}
	public NodoArbolBinario meterRec(NodoArbolBinario nodo,Object nodoInfo){
		if(nodo==null){
			return new NodoArbolBinario(nodoInfo);
		}
		if(menor(nodoInfo,nodo.getNodoInfo()))
			return nodo.getLeftChild().meterRec(nodo.getLeftChild(),nodoInfo);
	}
}
