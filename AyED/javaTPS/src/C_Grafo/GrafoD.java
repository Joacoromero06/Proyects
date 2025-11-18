package C_Grafo; 
import TDA.OperacionesGD; 
import TDA_Concretos.ListaDoubleComun;
import TDA_Concretos.ListaIntComun;

public class GrafoD extends Grafo implements OperacionesGD{
	protected MatrizGrafo matrizCostoF,matrizCaminoF;
	protected ListaIntComun listaCamino;
	protected ListaDoubleComun listaDistancia;

	public GrafoD(int ordenGrafo){
		super(ordenGrafo);
		this.listaCamino=new ListaIntComun();
		this.listaDistancia=new ListaDoubleComun();
		this.matrizCaminoF=new MatrizGrafo(this.ordenGrafo);
		this.matrizCostoF=new MatrizGrafo(this.ordenGrafo);
	}	
	public GrafoD(String archivo){
		super(archivo);
		this.listaCamino=new ListaIntComun();
		this.listaDistancia=new ListaDoubleComun();
		this.matrizCaminoF=new MatrizGrafo(this.ordenGrafo);
		this.matrizCostoF=new MatrizGrafo(this.ordenGrafo);
	}
	public String toString(){
		double currCost;
		String str="Mostrando Grafo Dirigido\n";
		for (int i=0; i<getOrden();i++){
			for (int j=0; j<getOrden();j++){				
				currCost=(double)this.matrizCosto.devolver(i, j);
				if (currCost!=infinito){
					str+="costo " + i + " a " + j + ": " + currCost;
				}								
			}
			str+="\n";			
		}	
		return str;	
	}

	public void muestraDijkstra(int startVertex){
		double currCost;
		Dijkstra(startVertex);
		for (int v=0; v<getOrden();v++)
			if (v!=startVertex){
				currCost=(double)this.listaDistancia.devolver(v);
				System.out.println("costo desde " + startVertex + " -> " + v + ": " + currCost);
				System.out.println("mostrando un camino desde "+ startVertex + " a " + v);
			
				System.out.print(startVertex);CaminoDijkstra(v);System.out.println();
			}			
	}
	public void Dijkstra(int startVertex){
		double minCost, currCost, arcCost; int minVertex, vertex;
		ListaIntComun listaSolucion=new ListaIntComun();
		for (int i=0; i<getOrden();i++){//Inicializa Listas Distancia, Camino, Solucion		
			listaSolucion.insertar(-1, i);
			if(i!=startVertex){
				this.listaCamino.insertar(startVertex,i);
				this.listaDistancia.insertar(this.matrizCosto.devolver(startVertex, i), i);
			}
			else{
				this.listaCamino.insertar(-1,i);
				this.listaDistancia.insertar(0, i);
			}
		}
		listaSolucion.reemplazar(startVertex,startVertex); // el primer vertice del camino
		
		for (int i=1; i<getOrden();i++){
			minCost=infinito;
			minVertex=-1;
			for (int w=0; w<getOrden();w++){//elige el vertice con menor distancia y no visitrado(avido) actualiza Solucion
				if (w!=startVertex){
					currCost=(double) this.listaDistancia.devolver(w);// 
					vertex=(int) listaSolucion.devolver(w);
					if (currCost<minCost && vertex==-1){
						minCost=currCost; minVertex=w;
					}
				}
			}
			if(minVertex!=-1){//ya elegimos el mejor ahora actualizamos Distancia y Camino a partir de el
				listaSolucion.reemplazar(minVertex, minVertex);
				this.listaDistancia.reemplazar(minCost, minVertex);
				for (int v=0;v<getOrden();v++){
					vertex=(int)listaSolucion.devolver(v);
					if (vertex==-1){
						arcCost=(double)this.matrizCosto.devolver(minVertex, v);
						currCost=(double)this.listaDistancia.devolver(v);
						if (minCost+arcCost<currCost){
							this.listaDistancia.reemplazar(minCost+arcCost, v);
							this.listaCamino.reemplazar(minVertex, v);					
						}					
					}
				}
			}
		}		
	}
	public void CaminoDijkstra(int i){
		int anterior=(int)this.listaCamino.devolver(i);
		if(anterior!=-1){
			CaminoDijkstra(anterior);
			System.out.print(i);
		}
		
	}

	public void muestraFloyd(){ 
		Floyd();
		System.out.println("Floyd: ");
		for(int i=0;i<ordenGrafo;i++){
			for(int j=0;j<ordenGrafo;j++){
				if(i!=j){
					if((double)matrizCostoF.devolver(i, j)!=infinito)
						System.out.println("Costo minimo de "+i+" hasta "+j+": "+(double)matrizCostoF.devolver(i, j));
				}
				
				
			}
		}
	}
	public void Floyd(){
		for(int i=0;i<ordenGrafo;i++)
			for(int j=0;j<ordenGrafo;j++){
				if(i!=j){
					this.matrizCostoF.actualizar((double)matrizCosto.devolver(i, j), i, j);
					this.matrizCaminoF.actualizar(-1, i, j);
				}	
				else{
					this.matrizCostoF.actualizar((double)0, i, j);
					this.matrizCaminoF.actualizar(-1, i, j);
				}	
			}	
		for(int k=0;k<ordenGrafo;k++)
			for(int i=0;i<ordenGrafo;i++)
				for(int j=0;j<ordenGrafo;j++)
					if((double)matrizCostoF.devolver(i,k)+(double)matrizCostoF.devolver(k,j)<(double)matrizCostoF.devolver(i,j)){
						this.matrizCostoF.actualizar((double)matrizCostoF.devolver(i,k)+(double)matrizCostoF.devolver(k,j), i, j);
						this.matrizCaminoF.actualizar(k, i, j);
					}
	}
	public void muestraCaminoFloyd(int origen, int destino){
		if((double)this.matrizCostoF.devolver(origen, destino)!=infinito) {
			System.out.print("Camino entre "+origen+" y "+destino+": ");
			System.out.print(origen);
			CaminoFloyd(origen,destino);
			System.out.print(" "+destino); 
		}else {
			System.out.println("NO hay Camino entre " + origen + " y " + destino);
		}	
	}
	private void CaminoFloyd(int i, int j){ 
		int k=(int)this.matrizCaminoF.devolver(i, j);
		if(k!=-1){
			CaminoFloyd(i,k);
			System.out.print(" "+k);
			CaminoFloyd(k,j);
		}
	}
	
}
