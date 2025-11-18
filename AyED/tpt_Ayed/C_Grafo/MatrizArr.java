package C_Grafo;

public class MatrizArr {
	protected Object[][] matriz;
	protected int nroFilas, nroColumnas;
	
	public MatrizArr(int nroFilas, int nroColumnas){
		this.nroFilas=nroFilas;
		this.nroColumnas=nroColumnas;
		this.matriz=new Object[this.nroFilas][this.nroColumnas];		
	}
	public int getNroFilas(){ return this.nroFilas;}
	public int getNroColumnas(){ return this.nroColumnas;}
	public void limpiaMatriz(){
		for (int i=0;i<getNroFilas();i++){
			for (int j=0;j<getNroColumnas();j++){
				this.matriz[i][j]=null;
			}
		}
	}	
	
	public void actualizar(Object objElemento, int posicionFila, int posicionColumna){
		if (posicionFila>=getNroFilas() || posicionFila<0){
				System.out.println("Error actualiza. Posicion fila inexistente ");
			}else{
				if (posicionColumna>=getNroColumnas() || posicionColumna<0){
					System.out.println("Error actualiza. Posicion columna inexistente ");
				}else{
					this.matriz[posicionFila][posicionColumna]=objElemento;
				}				
			}
		}
	public Object devolver(int posicionFila, int posicionColumna){
		Object objElemento=null;

		if (posicionFila>=getNroFilas() || posicionFila<0){
			System.out.println("Error devuelve. Posicion fila inexistente ");
		}else{
			if (posicionColumna>=getNroColumnas() || posicionColumna<0){
				System.out.println("Error devuelve. Posicion columna inexistente ");
			}else{
				objElemento = this.matriz[posicionFila][posicionColumna];
			}				
		}		
		return objElemento;
	}	
	
	public MatrizArr clonar(){
		MatrizArr clon=new MatrizArr(this.nroFilas,this.nroColumnas);
		for(int i=0;i<this.nroFilas;i++)
			for(int j=0;j<this.nroColumnas;j++)
				clon.actualizar(devolver(i, j),i,j);
		return clon;
	}
	
	public void triangSup() {
    for (int i = 0; i < this.nroFilas; i++) {
        for (int j = i+1; j < this.nroColumnas; j++) {  
            actualizar(devolver(i, j), j, i);  
        }
    }
}

	public void triangInf() {
    for (int i = 0; i < this.nroFilas; i++) {
        for (int j = 0; j < i; j++) {  
            actualizar(devolver(i, j), j, i); 
        }
    }
}
}
