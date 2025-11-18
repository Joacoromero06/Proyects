package C_Grafo;
import TDA_Concretos.IntegerSet;
import TDA_Concretos.ListaAristas;
import TDA_Concretos.ListaDoubleComun;
import TDA_Concretos.ListaIntComun;
import TDA_Concretos.ListaIntSetComun;
 
import TDA.OperacionesGND;


public class GrafoND extends Grafo implements OperacionesGND{	

	public GrafoND(int ordenGrafo) {
		super(ordenGrafo); 
	}
	public GrafoND(String archivo) {
		super(archivo); 
	}

	public String toString(){
		double currCost;
		String str="Mostrando Grafo no Dirigido\n";
		for (int i=0; i<getOrden();i++){
			for (int j=i+1; j<getOrden();j++){				
				currCost=(double)this.matrizCosto.devolver(i, j);
				if (currCost!=infinito){
					str+="costo " + i + " a " + j + "->" + currCost;
				}								
			}		
			str+="\n";	
		}		
		return str;
	}
	
	public ListaAristas Prim(int vertex){
		ListaDoubleComun MenorCosto = new ListaDoubleComun();
		ListaIntComun MasCercano = new ListaIntComun();
		ListaAristas Tree=new ListaAristas();
		double minCost, currCost, newCost; int minVertex, w;
		
		for (int i=0;i<getOrden();i++){//inicializacion prim
			MenorCosto.insertar(this.matrizCosto.devolver(vertex, i), i);
			MasCercano.insertar(vertex, i);
		}
		MasCercano.reemplazar(-1, vertex);MenorCosto.reemplazar(infinito, vertex);
	
		for (int i=0;i<getOrden()-1;i++){//ejecuta n-1 veces
			minCost=infinito;
			minVertex=-1;
			for (int j=0;j<getOrden();j++){//busca vertice que no este en el arbol y sea la de menor costo
				if ((int)MasCercano.devolver(j)!=(-1)&& (double)MenorCosto.devolver(j)!=infinito){//no esta en el subarbol
					currCost=(double)MenorCosto.devolver(j);
					if (currCost<minCost){
						minCost=currCost;
						minVertex=j;
					}
				}
			}
			//Agrego arista al arbol y indico que minvertex forma parte de Tree
			Tree.insertar(new Connection((int)MasCercano.devolver(minVertex), minVertex, minCost), i);
			MenorCosto.reemplazar(infinito, minVertex);
			MasCercano.reemplazar(-1,minVertex);

			for (int j=0;j<getOrden();j++){//Actualizo costos con el nuevo minvertex
				if (j!=minVertex){ 
					w=(int)MasCercano.devolver(j);
					currCost=(double)MenorCosto.devolver(j);
					newCost=(double)this.matrizCosto.devolver(minVertex, j);
					if (newCost< currCost&& w!=-1){
						MenorCosto.reemplazar(newCost, j);
						MasCercano.reemplazar(minVertex, j);
					}
				}
			}
		}
		return Tree;		
	}
	
	public ListaAristas Kruskal(){
		double currCost; int counter; int k, posI, posJ; boolean flag;
		Connection conexion;
		GraphPriorityQueue colaP=new GraphPriorityQueue();
		IntegerSet conjuntoE,conjuntoU;
		ListaIntSetComun NodosArbol = new ListaIntSetComun();
		ListaAristas Tree=new ListaAristas();
		
		for (int i=0;i<getOrden();i++){//Inicia lista de subarboles minimos
			conjuntoE = new IntegerSet();
			conjuntoE.meter(i);
			NodosArbol.insertar(conjuntoE, i);			
		}
		
		for (int i=0; i<getOrden();i++){//Ordena las aristas segun sus pesos y orden secuencial
			for (int j=i+1;j<getOrden();j++){
				currCost=(double)this.matrizCosto.devolver(i, j);
				if (currCost!=infinito){
					colaP.meter(new Connection(i, j, currCost));
				}
			}
		}
		
		counter=0;
		while (counter<getOrden()-1){//realiza n-1 uniones de subarboles abarcadores minimos que no generen ciclos
			conexion=(Connection)colaP.sacar();  
			k=0; flag=true;
			posI=posJ=-1;
			while (k<=NodosArbol.tamanio()-1 && flag){//determina en que conjunto(subarbol cubridor) estan el vertice i y j de la conexion con menor peso
				conjuntoE=(IntegerSet)NodosArbol.devolver(k); 
				if (conjuntoE.pertenece(conexion.getVertexI()))
					posI=k;
				if (conjuntoE.pertenece(conexion.getVertexJ()))
					posJ=k;
				if (posI!=-1 && posJ!=-1)
					flag=false;
				else
					k++;
				
			}
			
			if (!flag && posI!=posJ){//si los vertices i y j son de dos subarboles distintos su arista no forma un ciclo
				Tree.insertar(conexion,counter );
				conjuntoU = new IntegerSet();
				conjuntoU.union((IntegerSet)NodosArbol.devolver(posI), (IntegerSet)NodosArbol.devolver(posJ));
				NodosArbol.reemplazar(conjuntoU, posI);
				NodosArbol.eliminar(posJ);
				counter++;	 			
			}			
		}
		return Tree;
	}
	 
}
