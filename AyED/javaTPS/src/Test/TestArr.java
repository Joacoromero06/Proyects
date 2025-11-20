package Test;
import CL1.*;
import assets.Entrada;

public class TestArr {
	public static void main(String[] args) {
		
		//TEST PILA_ARR
		int n;	
		System.out.println("Ingrese cantidad");
		n = Entrada.sc.nextInt();
		if (n > 0) {
			PilaArr objPila = new PilaArr(n);
			for (int i=0; i<n; i++) {//Carga
				System.out.println("Ingrese elemento");
				objPila.meter(Entrada.sc.nextInt());				
			}
			
			while (!objPila.estaVacia()) {//Muesta	
				System.out.println("Elemento Pila " + objPila.sacar());		
			}
			objPila.sacar();//Ya sacamos todo no va a dejar sacar, en objaux quedo null
		}
		
		//TEST COLA ARR
		ColaArr objCola = new ColaArr(5);
						
		objCola.meter(10);objCola.meter(20);objCola.meter(30);objCola.meter(40);
		objCola.meter(50);//No va a dejar meter la colaarr tiene restringido un elemento del vector
		objCola.meter(60);

		while (!objCola.estaVacia()) {//Muestra
			System.out.println("Elemento Cola " + objCola.sacar());
			
		}		
		
	}

}
