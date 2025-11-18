package CL1;
import TDA.*;

//IMPLEMENTACION DE CL1->COLA DE OBJETOS CON ARRAY
public class ColaArr implements OperacionesCL1, OperacionesCL1Estaticos{
	private Object[] cola;
	private int finalC, frenteC;//Manjeo la cola con dos indices para colacircular
	private int tamCola;
	//indice finalC tenga el ultimo de la cola(para meter incremento y meto)(ojo es circular)Si no esta llena
	//indice freneC tengo el centinela anterior al primero en la cola 

	public ColaArr(int tamCola) {
		this.tamCola = tamCola;
		this.cola = new Object[this.tamCola];
		limpiar(); 
	}

	public void limpiar() {//Los indices empiezan en el ultimo elemento del array
		this.finalC = this.frenteC = tamCola - 1;
	}

	public void meter(Object elemento) {
		if (!estaLlena()) {
			if (this.finalC == this.tamCola - 1){//Si final esta en el ult elemento del vector, "da vuelta"
				this.finalC = 0;
			}else{
				incrementaFinal(); //Sino  incremento normal y agrego
			}
			this.cola[this.finalC] = elemento;//Incremento y meto para sobreescribir basura de memoria que queda cuando saco
		} else {
				System.out.println("Error. Cola llena...");
			}
	}
	
	
	public boolean estaLlena() {//Como dejo un elemento que uso para manejar el array circular
		int p;//El frente esta en ese elemento y adelante de frente esta el primero en la cola
		if (this.finalC == this.tamCola - 1){
			p = 0;//Hago que p guarde la posicion siguiente a finalC 
		}else{
			p = this.finalC + 1;
		}//Porque como meto en la posicion siguiente antes chequeo si esa posicion siguiente es donde esta frente
		//osea una anterior al primero en la cola, osea el centinela
		return (p == this.frenteC);//Cuando el siguiente al final dio vuelta y llego al frente
	}
		
	public Object sacar() {
		Object elemento = null;
		if (!estaVacia()){
			if (this.frenteC == this.tamCola - 1){
				this.frenteC = 0;
			}else{
				incrementaFrente();//Como frente esta antes del primero en la cola primero lo adelanto
			}//Despues guardo el valor del primero en la cola.//SIGUE ESTANDO AHI se sobrescribira por eso incremento y meto
			elemento = this.cola[this.frenteC];
		} else {
			System.out.println("Error. Cola vacia...");
		}
		return elemento;
	}	
	
	public boolean estaVacia() {//Cuando saque tanto que frente se incremento guardo y dejo como basura el elemento donde esta
		return frenteC==finalC;//Que alcanzo a final, al sacar frente persigue a final, final al meter intentara pero no lo va alcanzar
	}
	
	private void incrementaFinal() { this.finalC++; }
	private void incrementaFrente() { this.frenteC++; }	
	
	
}