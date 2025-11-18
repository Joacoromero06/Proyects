package C_Grafo;
import CL1.ColaPrioridadSLL;
import TDA.Nodo;

public class GraphPriorityQueue extends ColaPrioridadSLL {
	
	
	public boolean esMenor(Object objA, Object objB){
		return ((Connection)objA).getConnectionCost()<((Connection)objB).getConnectionCost();
	}
	public boolean esMayor(Object objA, Object objB){
		return ((Connection)objA).getConnectionCost()>((Connection)objB).getConnectionCost();
	}
	public boolean iguales(Object objA, Object objB){
		return ((Connection)objA).getConnectionCost()==((Connection)objB).getConnectionCost();
	}
	
	void muestra(){
		Nodo temp; Connection conexion;		
		if (!estaVacia()){
			temp=this.frenteC;
			while (temp!=null){
				conexion=(Connection)temp.getNodoInfo();
				System.out.println("Conexion " + conexion.getVertexI() + " -> " + conexion.getVertexJ() + ":" + conexion.getConnectionCost());
				temp=temp.getNextNodo();
			}			
		}else{
			System.out.println("Error muestra. Cola vacia");
		}		
	}	
}
