
package C_Arbol;

public class NodoArbolBinario {
	private Object nodoInfo;
	private NodoArbolBinario leftChild, rightChild;
	
	public NodoArbolBinario(){
		setNodoInfo(null);
		setLeftChild(null);
		setRightChild(null);
	}
	public NodoArbolBinario(Object nodoInfo){
		this(nodoInfo, null,null);
	}	
	public NodoArbolBinario(Object nodoInfo, NodoArbolBinario leftChild, NodoArbolBinario rightChild){
		setNodoInfo(nodoInfo);
		setLeftChild(leftChild);
		setRightChild(rightChild);
	}
	
	public void setNodoInfo(Object nodoInfo){
		this.nodoInfo=nodoInfo;
	}
	public Object getNodoInfo(){
		return this.nodoInfo;
	}
	public void setLeftChild(NodoArbolBinario leftChild){
		this.leftChild=leftChild;
	}
	public void setRightChild(NodoArbolBinario rightChild){
		this.rightChild=rightChild;
	}
	public NodoArbolBinario getLeftChild(){
		return this.leftChild;
	}
	public NodoArbolBinario getRightChild(){
		return this.rightChild;
	}	
}
