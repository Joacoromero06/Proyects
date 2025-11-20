public class ListaSumadores{
    SumaParcial[] lista;
    Matriz matriz;
    int n;
    double suma;

    public ListaSumadores(int n, Matriz matriz){
        this.lista = new SumaParcial[n];
        this.matriz = matriz;
        this.n = n;
        inicializarHilos();
    }
    public void inicializarHilos(){
        int i;
        int col_ini, col_fin;
        int h;

        // Computo la cantidad de columnas promedio por hilo
        h = this.matriz.getColumnas() / this.n;

        // Creacion de cada uno de los hilos
        for(i = 0; i < this.n; i++){
            col_ini = h*i;
            col_fin = h*(i+1);

            if(i == this.n -1) col_fin = this.matriz.getColumnas(); 

            this.lista[i] = new SumaParcial(this.matriz, col_ini, col_fin);

            //aca start ?
        }
    }

    public void sumar(){
        int i;
        long t_ini, t_fin, t_total;
        double suma;

        t_ini = System.currentTimeMillis();
        for(i = 0; i < this.n; i++){
            this.lista[i].start();
        }
        try{
            for(i = 0; i < this.n; i++){
                this.lista[i].join();
            }
        }catch(Exception e){
            System.out.println("El error es" + e + " printear e.getmessage()");
        }
        t_fin = System.currentTimeMillis();
        t_total = t_fin - t_ini;

        suma = 0;
        for(i = 0; i < this.n; i++){
            suma += this.lista[i].getAcum();
        }

        System.out.println("La suma es: " + suma);
    }
}