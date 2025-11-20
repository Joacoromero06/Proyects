public class Hilos extends Thread {
    private Imagen imagen;
    private int Ci,Cf;
    private Imagen rdo;
   
    public Hilos(Imagen imagen, int Ci, int Cf, Imagen rto){
        this.imagen=imagen;
        this.Ci=Ci;
        this.Cf=Cf;
        this.rdo = rto;
    }
    

    public void run(){
        int i,j; 
		for (i = 0; i < this.imagen.getFilas(); i+=2) {
		    for (j = this.Ci; j < this.Cf; j+=2) {
		    	int max = imagen.devolver(i, j); // hay que hacer devolver 
		    	if (i + 1 < imagen.getFilas()) max = Math.max(max, imagen.devolver(i+1, j));
	            if (j + 1 < Cf) max = Math.max(max, imagen.devolver(i, j+1));
	            if (i + 1 < imagen.getFilas() && j + 1 < Cf) max = Math.max(max, imagen.devolver(i+1, j+1));

                System.out.println("soy un hilo, actualizo: " + max +  "en posicion: " + i/2 + " " + j/2);

	            rdo.actualizar(max,i/2, j/2); // hay que hacer actualizar

	        }
		} 
    }

    public Imagen getRdo(){
        return this.rdo;
    }
        
}


/*  IMAGEN ORIGEN dividad con 2 hilos
            [1,2,   3,4    ,5,6]                   
            [1,2,   3,4    ,5,6]
if((columna/n)%2!=0){
    columnasRto+1;
}

            [1,2,3,4, 5,6]
            [1,2,3,4, 5,6]
            
            [1,2,3, 4, 5,6]
            [1,2,3,4, 5,6]
            
            [1,2,3,4, 5,6]

            [ 2, 4, 6 ]
            [ 2, 3,  4, 6 ]
            [ 2, 3,  4, 6 ]
            [ 2, 3,  4, 6 ]
            i_rto = 0 
            hilo1: j_rto = Ci;
            hilo2: j_rto = Ci / 2 if (Ci % 2 == 0) else Ci / 2 + 1;
            
            IMAGEN RTO 2 hilos  j_rdo= Cf/2
            hilo1: j_rto = 0; i_rto = 0;
            hilo2:
            if( getColumnas() % 2 == 0 ) 
                division = getColumnas() / 2;
            else
                division = getColumas() / 2 + 1; 
            for(){
            if(i<n-1)
            hilos[i]=new Hilos(imagen,i*division, (i+1)*division,)
            else
            hilos[i]=new Hilos(imagen,i*division, getColumnas(),)
            }
            i_rto = 0;                     
            if(j_rto<2)
             error;
            i_rto = 0;
        */