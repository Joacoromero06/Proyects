package TDA;

public class NodoDoble {
	private Object nodoInfo;
	private NodoDoble prevNodo, nextNodo;
	
	public NodoDoble(Object nodoInfo){
		this(nodoInfo,null,null);
	} 
	
	public NodoDoble(Object nodoInfo, NodoDoble nextNodo){
		this(nodoInfo,null,nextNodo);
	} 
	
	public NodoDoble(Object nodoInfo, NodoDoble prevNodo, NodoDoble nextNodo){
		this.nodoInfo=nodoInfo;
		this.prevNodo=prevNodo; this.nextNodo=nextNodo; 
	}
	
	public void setPrevNodo(NodoDoble prevNodo){
		this.prevNodo=prevNodo;
	}
	
	public NodoDoble getPrevNodo(){
		return this.prevNodo; 
	}
	
	public void setNextNodo(NodoDoble nextNodo){
		this.nextNodo=nextNodo;
	}
	
	public NodoDoble getNextNodo(){
		return this.nextNodo; 
	}
	public void setNodoInfo(Object nodoInfo){
		this.nodoInfo=nodoInfo; 
	}
	public Object getNodoInfo(){
		return this.nodoInfo;
	}

}
