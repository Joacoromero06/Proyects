package CL2;
import TDA.Nodo;
import TDA.OperacionesConj;
public abstract class AbsSet implements OperacionesConj{
	protected Nodo conjunto;
	protected int cardinal;
	
	public AbsSet(){
		vaciar();		
	}
	public void vaciar(){
		this.conjunto=null;
		this.cardinal=0;
	}
	public boolean estaVacio(){
		return (this.conjunto==null);}
	public int getCardinal(){
		return this.cardinal;}
	public abstract boolean iguales(Object objA, Object objB);
	
	public boolean pertenece(Object objElemento){
		Nodo temp; boolean flag;
		boolean response=false;
		if (!estaVacio()){
			temp=this.conjunto;
			flag=false;
			while (temp!=null && !flag){
				if (!iguales(objElemento,temp.getNodoInfo())){
					temp=temp.getNextNodo();
				}else{
					flag=true;
				}
			}
			if (flag){
				response=true;
			}	
		}else{
			//System.out.println("error esta. Conjunto vacio..");
		}		
		return response;
	}
	public void meter(Object objElemento){
		if (!pertenece(objElemento)){
			this.conjunto=new Nodo(objElemento, this.conjunto);
			this.cardinal++;
		}else{
			System.out.println("error meter. El elemento ya esta en conjunto..");
		}		
	}	
	public void sacar(Object objElemento){
		Nodo temp;
		boolean flag;
		
		if (!estaVacio()){
			if (pertenece(objElemento)){
				temp=this.conjunto;
				if (iguales(objElemento,temp.getNodoInfo())){
					this.conjunto=this.conjunto.getNextNodo();
				}else{
					flag=false;
					while (temp.getNextNodo()!=null && !flag){
						if (!iguales(objElemento,temp.getNextNodo().getNodoInfo())){
							temp=temp.getNextNodo();
						}else{
							flag=true;
						}
					}
					temp.setNextNodo(temp.getNextNodo().getNextNodo());					
				}
				this.cardinal--;
				
				if (estaVacio()){
					vaciar();
				}
				
			}else{
				System.out.println("Error sacar. El elemento no esta.");
			}
		}else{
			System.out.println("Error sacar. Conjunto vacio");			
		}		
	}
	
	private Object sacarPrimero(){
		Object objElemento=null;
		if (!estaVacio()){
			objElemento=this.conjunto.getNodoInfo();
			this.conjunto=this.conjunto.getNextNodo();
			
			if (estaVacio()){
				vaciar();
			}
		}else{
			//System.out.println("Error sacar. Conjunto vacio");			
		}
		return objElemento;		
	}	
	public void union(AbsSet conjA, AbsSet conjB){
		Object objElemento;
		
		if (!conjA.estaVacio() || !conjB.estaVacio()){
			vaciar();
			while (!conjA.estaVacio()){
				objElemento=conjA.sacarPrimero();		
				meter(objElemento);				
			}	
			//System.out.println("ya esta conjA");
			
			while (!conjB.estaVacio()){
				objElemento=conjB.sacarPrimero();
				if (!pertenece(objElemento)){
					meter(objElemento);	
				}				
			}			
		}else{
			//System.out.println("Error union. Conjuntos vacios");
		}		
	}

}

	
	

