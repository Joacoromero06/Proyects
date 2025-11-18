package C_Grafo;

public class Connection {
	protected int vertexI, vertexJ;
	protected double connectionCost;
	
	public Connection(int vertexI, int vertexJ, double connectionCost){
		this.vertexI=vertexI;
		this.vertexJ=vertexJ;
		this.connectionCost=connectionCost;
	}
	public int getVertexI(){
		return this.vertexI;
	}
	public int getVertexJ(){
		return this.vertexJ;
	}
	public double getConnectionCost(){
		return this.connectionCost;
	}	
	public String toString(){
		return "Arista: "+this.vertexI+"->"+this.vertexJ+" peso: "+this.connectionCost;
	}
}
