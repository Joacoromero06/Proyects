package TDA;

import CL2.AbsSet;

public interface OperacionesConj {
	
	public boolean estaVacio();
	public void vaciar();
	public int getCardinal();
	public boolean pertenece(Object objElemento);
	public void meter(Object objElemento);
	public void sacar(Object objElemento);
	public void union(AbsSet conjA, AbsSet conjB);
	

}
