package C_Grafo;

public class MatrizGrafo extends MatrizArr{
	public MatrizGrafo(int ordenGrafo){
		super(ordenGrafo, ordenGrafo);
	}
	
	public boolean areConnected(int i, int j){
		boolean response=false;
		if (i>=0 && i<getNroFilas() && j>=0 && j<getNroColumnas()){
			if (this.matriz[i][j]!=null){
				response=true;
			}
		}				
		return response;
	}
	public MatrizGrafo clonar(){
		MatrizGrafo clon=new MatrizGrafo(this.nroFilas);
		for(int i=0;i<this.nroFilas;i++)
			for(int j=0;j<this.nroColumnas;j++)
				clon.actualizar(devolver(i, j),i,j);
		return clon;
	} 
}
