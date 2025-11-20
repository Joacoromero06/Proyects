package C_Grafo;
import TDA.OperacionesG;
import TDA_Concretos.ListaBoolean;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import CL1.ColaSLinkedList;

public class Grafo implements OperacionesG{
	protected MatrizGrafo matrizCosto;
	protected int ordenGrafo;
	protected static double infinito=10000;
	
	public Grafo(int ordenGrafo) {
		this.ordenGrafo = ordenGrafo;
		this.matrizCosto = new MatrizGrafo(getOrden());
	}
	public Grafo(String archivo){
		this(orden(archivo));
		cargarGrafo(archivo);
	}
	private void cargarGrafo(String nombreArchivo) {
		try (Scanner scanner = new Scanner(new File(nombreArchivo))) {
			int orden = scanner.nextInt(); 
			for (int i = 0; i < orden; i++) 
				for (int j = 0; j < orden; j++) 
					matrizCosto.actualizar(infinito, i, j);
			while (scanner.hasNextInt()) {
				int i = scanner.nextInt();
				int j = scanner.nextInt();
				double costo = scanner.nextDouble();
				matrizCosto.actualizar(costo, i, j);
				matrizCosto.actualizar(costo, j, i); // simetría
			}
		} catch (FileNotFoundException e) {
			System.err.println("Archivo no encontrado: " + nombreArchivo);
		}
	}
	private static int orden(String nombreArchivo) {
		try (Scanner scanner = new Scanner(new File(nombreArchivo))) {
            return scanner.nextInt();
        } catch (FileNotFoundException e) {
            throw new RuntimeException("No se pudo abrir el archivo: " + nombreArchivo);
        }
	}
	public int getOrden(){
		return this.ordenGrafo;
	}
 
	private void bpf(ListaBoolean listaMarca, int v){
		boolean marcado;
		double currCost;

		listaMarca.reemplazar(true, v);//v es el inicio
		System.out.println("vertice "+ v);
		for (int w=0;w<getOrden();w++){//recorre todos los nodos y pregunta para v
			marcado=(boolean)listaMarca.devolver(w);
			currCost=(double)this.matrizCosto.devolver(v,w);
			if (currCost!=infinito && !marcado){//si no fue visitado y es adyacente
				bpf(listaMarca,w);//tenemos que mostrarlo y hacer lo mismo
			}
		}
	}
	public void muestraBPF(){
		ListaBoolean listaMarca = new ListaBoolean();
		boolean marcado;
		for (int v=0;v<getOrden();v++){
			listaMarca.insertar(false, v);
		}
		for (int v=0;v<getOrden();v++){//puede que haya conj de nodos no conexos
			marcado=(boolean)listaMarca.devolver(v);
			if (!marcado){
				bpf(listaMarca,v);
			}
		}		
	}
	
	private void bea(ListaBoolean listaMarca, int v){
		boolean marcado;
		double currCost;
		int w;
		ColaSLinkedList cola = new ColaSLinkedList();
		
		listaMarca.reemplazar(true, v);
		System.out.println("vertice "+ v);
		cola.meter(v);
		
		while (!cola.estaVacia()){
			w=(int)cola.sacar();//w nodo primero en orden de visita
			for (int z=0;z<getOrden();z++){//recorre todos los nodos y pregunta para w
				marcado=(boolean)listaMarca.devolver(z);
				currCost=(double)this.matrizCosto.devolver(w,z);
				if (currCost!=infinito && !marcado){//si no fue visitado y es adyacente
					listaMarca.reemplazar(true, z);
					cola.meter(z);
					System.out.println("arista visitada " + w + " -> " + z);
				}		
			}
		}
	}
	public void muestraBEA(){
		ListaBoolean listaMarca = new ListaBoolean();
		boolean marcado;
		for (int v=0;v<getOrden();v++){
			listaMarca.insertar(false, v);
		}
		for (int v=0;v<getOrden();v++){
			marcado=(boolean)listaMarca.devolver(v);
			if (!marcado){
				bea(listaMarca,v);
			}
		}		
	}
}
