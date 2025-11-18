package C_Grafo; 
import TDA_Concretos.NumReal;
import TDA.OperacionesGD;
import TDA_Concretos.ListaBoolean;
import TDA_Concretos.ListaDoubleComun;
import TDA_Concretos.ListaIntComun;

public class GrafoD extends Grafo implements OperacionesGD{
	protected MatrizGrafo matrizCostoF,matrizCaminoF;
	protected ListaIntComun listaCaminoD;
	protected ListaDoubleComun listaDistanciaD;
	protected ListaIntComun rutaHamiltoniana;
	protected MatrizGrafo matrizAux;

	public GrafoD(int ordenGrafo){ 
		super(ordenGrafo);
		this.listaCaminoD=new ListaIntComun();
		this.listaDistanciaD=new ListaDoubleComun();
		this.matrizCaminoF=new MatrizGrafo(this.ordenGrafo);
		this.matrizCostoF=new MatrizGrafo(this.ordenGrafo);
		this.rutaHamiltoniana=new ListaIntComun();
	}	
	public GrafoD(String archivo){
		super(archivo);
		this.listaCaminoD=new ListaIntComun();
		this.listaDistanciaD=new ListaDoubleComun();
		this.matrizCaminoF=new MatrizGrafo(this.ordenGrafo);
		this.matrizCostoF=new MatrizGrafo(this.ordenGrafo);
		this.rutaHamiltoniana=new ListaIntComun();
		this.matrizAux=this.matrizCosto.clonar();
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
				currCost=(double)this.listaDistanciaD.devolver(v);
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
				this.listaCaminoD.insertar(startVertex,i);
				this.listaDistanciaD.insertar(this.matrizCosto.devolver(startVertex, i), i);
			}
			else{
				this.listaCaminoD.insertar(-1,i);
				this.listaDistanciaD.insertar(0, i);
			}
		}
		listaSolucion.reemplazar(startVertex,startVertex); // el primer vertice del camino
		
		for (int i=1; i<getOrden();i++){
			minCost=infinito;
			minVertex=-1;
			for (int w=0; w<getOrden();w++){//elige el vertice con menor distancia y no visitrado(avido) actualiza Solucion
				if (w!=startVertex){
					currCost=(double) this.listaDistanciaD.devolver(w);// 
					vertex=(int) listaSolucion.devolver(w);
					if (currCost<minCost && vertex==-1){
						minCost=currCost; minVertex=w;
					}
				}
			}
			if(minVertex!=-1){//ya elegimos el mejor ahora actualizamos Distancia y Camino a partir de el
				listaSolucion.reemplazar(minVertex, minVertex);
				this.listaDistanciaD.reemplazar(minCost, minVertex);
				for (int v=0;v<getOrden();v++){
					vertex=(int)listaSolucion.devolver(v);
					if (vertex==-1){
						arcCost=(double)this.matrizCosto.devolver(minVertex, v);
						currCost=(double)this.listaDistanciaD.devolver(v);
						if (minCost+arcCost<currCost){
							this.listaDistanciaD.reemplazar(minCost+arcCost, v);
							this.listaCaminoD.reemplazar(minVertex, v);					
						}					
					}
				}
			}
		}		
	}
	public void CaminoDijkstra(int i){
		int anterior=(int)this.listaCaminoD.devolver(i);
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
	
	public void Dijkstra2(int startVertex,MatrizGrafo matriz){//intento de parametrizacion mala practica ropme paradigmaPOO
		double minCost, currCost, arcCost; int minVertex, vertex;
		ListaIntComun listaSolucion=new ListaIntComun(); 
		for (int i=0; i<getOrden();i++){//Inicializa Listas Distancia, Camino, Solucion		
			listaSolucion.insertar(-1, i);
			if(i!=startVertex){
				this.listaCaminoD.insertar(startVertex,i); 
				this.listaDistanciaD.insertar(matriz.devolver(startVertex, i), i);
			}
			else{
				this.listaCaminoD.insertar(-1,i);
				this.listaDistanciaD.insertar(0, i);
			}
		} 
		listaSolucion.reemplazar(startVertex,startVertex); // el primer vertice del camino
		for (int i=1; i<getOrden();i++){
			minCost=infinito;
			minVertex=-1;
			for (int w=0; w<getOrden();w++){//elige el vertice con menor distancia y no visitrado(avido) actualiza Solucion
				if (w!=startVertex){
					currCost=(double) this.listaDistanciaD.devolver(w);// 
					vertex=(int) listaSolucion.devolver(w);
					if (currCost<minCost && vertex==-1){
						minCost=currCost; minVertex=w;
					}
				}
			}
			if(minVertex!=-1){//ya elegimos el mejor ahora actualizamos Distancia y Camino a partir de el
				listaSolucion.reemplazar(minVertex, minVertex);
				this.listaDistanciaD.reemplazar(minCost, minVertex);
				for (int v=0;v<getOrden();v++){
					vertex=(int)listaSolucion.devolver(v);
					if (vertex==-1){
						arcCost=(double)matriz.devolver(minVertex, v);
						currCost=(double)this.listaDistanciaD.devolver(v);
						if (minCost+arcCost<currCost){
							this.listaDistanciaD.reemplazar(minCost+arcCost, v);
							this.listaCaminoD.reemplazar(minVertex, v);					
						}					
					}
				}
			}
		}	
	}
	public void CalculaRutaDijkstra(int origen,int destino){
		ListaIntComun ruta=new ListaIntComun();
		Dijkstra(origen);
		CaminoDijkstra(destino,ruta);
		ruta.insertar(destino, ruta.tamanio());
		this.rutaHamiltoniana=ruta.clonar();
	}
	public void CaminoDijkstra(int i,ListaIntComun ruta){
		int anterior=(int)this.listaCaminoD.devolver(i);
		if(anterior!=-1){
			CaminoDijkstra(anterior,ruta);
			ruta.insertar(anterior, ruta.tamanio());
		}
	}

	public void CalculaRutaHamiltoniana(int origen,int destino,int algoritmo){
		ListaIntComun camino=new ListaIntComun();
		ListaBoolean visitados=new ListaBoolean();
		for(int i=0;i<getOrden();i++)
			visitados.insertar(false, i);
		visitados.reemplazar(true, origen);
		camino.insertar(origen, camino.tamanio());
		switch (algoritmo) {//CALCULA CON:
			case 1://BT_1
				if(RutaHamiltoniana_BT1(origen, destino,visitados,camino))
					this.rutaHamiltoniana=camino.clonar();//o tambien puede ser: camino.copiarEn(this.rutaHamiltoniana);
				else
					this.rutaHamiltoniana.limpiar();
			break;
			case 2://BT_2 
				ListaIntComun mejorCamino=new ListaIntComun();
				NumReal menorCosto =new NumReal(infinito);
				if(RutaHamiltoniana_BT2(origen, destino, visitados, camino,0 ,mejorCamino ,menorCosto ))
					this.rutaHamiltoniana=mejorCamino.clonar();
				else
					this.rutaHamiltoniana.limpiar();
			break;
			default:
				break;
		}
	}
	public boolean RutaHamiltoniana_BT1(int actual,int destino,ListaBoolean visitados,ListaIntComun camino){
		if(camino.tamanio()==getOrden()&& actual==destino)
			return true;
		for(int i=0;i<getOrden();i++){
			if(!visitados.devolver(i)){
				visitados.reemplazar(true, i);
				camino.insertar(i, camino.tamanio());
				if(RutaHamiltoniana_BT1(i, destino, visitados, camino))
					return true;
				camino.eliminar(camino.tamanio()-1);
				visitados.reemplazar(false, i);
			}
		}
		return false;
	}
	public boolean RutaHamiltoniana_BT2(int actual,int destino,ListaBoolean visitados,ListaIntComun caminoActual, double costoActual,ListaIntComun mejorCamino, NumReal mejorCosto){
		boolean b=false;
		//System.out.println("\n\nNodo actual: " + actual + ", Camino: "+caminoActual+" La lista de visitados: "+ visitados);
		if(caminoActual.tamanio()==getOrden()&&actual==destino){
			if(costoActual<mejorCosto.getReal()){
				mejorCosto.setReal(costoActual);
				//System.out.println("El mejorCamino: "+mejorCamino+" es reemplazado por: "+caminoActual+" Costo:"+costoActual);
				mejorCamino.limpiar();
				ListaIntComun clon=caminoActual.clonar();
				clon.copiaEn(mejorCamino); 
			}	
			b=true;
		}
		double costArco; 
		for(int i=0;i<getOrden();i++){
			costArco=(double)this.matrizCosto.devolver(actual,i);//cuidado con lo que tiene matrizCosto, si tiene null hay que chequear eso
			if(!visitados.devolver(i)&&costArco!=infinito){
				visitados.reemplazar(true, i);
				caminoActual.insertar(i, caminoActual.tamanio()); 
				costoActual+=costArco;
				if(costoActual<mejorCosto.getReal())//poda aquellos camino que ya sabemos que van a ser peores
					if(RutaHamiltoniana_BT2(i, destino, visitados, caminoActual, costoActual, mejorCamino, mejorCosto))
						b= true;		
						
				visitados.reemplazar(false, i);
				caminoActual.eliminar(caminoActual.tamanio()-1);
				costoActual-=costArco;
			}
		}
		return b;
	}
	
	public void RutaHamiltoniana_Greedy1(int origen,int destino,ListaBoolean visitados,ListaIntComun camino){
		int greedyVertex=-1,actual;
		double menorCosto,costoActual=-1;
		visitados.reemplazar(true, origen);
		camino.insertar(origen, camino.tamanio());
		actual=origen;
		menorCosto=infinito;
		for(int i = 1 ; i<getOrden();i++){
			for(int k=0;k<getOrden();k++){
				costoActual=(double)this.matrizCosto.devolver(actual, k);
				if(costoActual<menorCosto&&!visitados.devolver(k)){
					menorCosto=costoActual;
					greedyVertex=actual;
				}
			}
			visitados.reemplazar(true, greedyVertex);
			camino.iguales(greedyVertex, camino.tamanio());
			actual=greedyVertex;
			menorCosto=infinito;
		}
	}
	public void RutaHamiltoniana_Greedy2(int origen,int destino,ListaBoolean visitados,ListaIntComun camino){
		int greedyVertex=-1,actual;
		double menorCosto,costoActual=-1;
		visitados.reemplazar(true, origen);
		camino.insertar(origen, camino.tamanio());
		actual=origen;
		menorCosto=infinito;
		for(int i = 1 ; i<getOrden();i++){
			for(int k=0;k<getOrden();k++){
				costoActual=(double)this.matrizCosto.devolver(actual, k);
				if(costoActual<menorCosto&&!visitados.devolver(k))
					if(k!=destino){
					menorCosto=costoActual;
					greedyVertex=actual;
					}
			}
			visitados.reemplazar(true, greedyVertex);
			camino.iguales(greedyVertex, camino.tamanio());
			actual=greedyVertex;
			menorCosto=infinito;
		}
	}
	
	public void usarTriangSup(){this.matrizCosto.triangSup();}
	public void usarTriangInf(){this.matrizCosto.triangInf();}
	public void usarTodaMatriz(){this.matrizCosto=this.matrizAux.clonar();}

	public MatrizGrafo getMatrizCostoF() {return matrizCostoF;}
	public MatrizGrafo getMatrizCaminoF() {return matrizCaminoF;}
	public ListaIntComun getListaCaminoD() {return listaCaminoD;}
	public ListaDoubleComun getListaDistanciaD() {return listaDistanciaD;}
	public ListaIntComun getRutaHamiltoniana() {return rutaHamiltoniana.clonar();}
	public MatrizGrafo getMatrizAux() {return matrizAux;}

}
