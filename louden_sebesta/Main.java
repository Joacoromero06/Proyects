public class Main{
    public static void main(String[] args){
        int m, n, ini, fin, n_hilos;
        m = 10000; //tamaño de filas
        n = 10000; // tamaño de columnas
        
        Matriz A = new Matriz(m, n, -10, 10);
        
        n_hilos = 10;
        ListaSumadores l = new ListaSumadores(n_hilos, A);
        l.sumar();
        //A.mostrar();
        
    }
    public static void sinHilos(Matriz A){
        long t_ini, t_fin, t_total;
        double total;
        
        t_ini = System.currentTimeMillis();
        total = A.sumaTotal();
        t_fin = System.currentTimeMillis();

        t_total = t_fin - t_ini;

        System.out.println("La suma total: " + total);
        System.out.println("La suma total sin hilos tardo: " + t_total);
    }
    public static void con2Hilos(Matriz A){
        long t_ini, t_fin, t_total;
        double total;

        SumaParcial h1, h2;
        h1 = new SumaParcial(A, 0, A.getColumnas()/2 + 1);
        h2 = new SumaParcial(A, A.getColumnas()/2 + 1, A.getColumnas());


        t_ini = System.currentTimeMillis();
        h1.start();
        h2.start();
        try{
            h1.join();
            h2.join();
        }catch(Exception e){
            System.out.println("hace get message de e");
        }
        t_fin = System.currentTimeMillis();

        t_total = t_fin - t_ini;
        total = h1.getAcum() + h2.getAcum();

        System.out.println("La suma total: " + total);
        System.out.println("La suma total con 2 hilos tardo: " + t_total);
    }
    public static void conNHilos(Matriz A, int n){
        long t_ini, t_fin, t_total;
        double total;
        int i;

        // construir vector de hilos
        SumaParcial[] hilos = new SumaParcial[n];

        // instanciar los n hilos con sus correspondientes sectores
        for(i = 0; i < n; i++){
            int col_ini = (i) * A.getColumnas() / n;
            int col_fin = (i+1) * A.getColumnas() / n;
            hilos[i] = new SumaParcial(A, col_ini, col_fin);
        }

        // Computar la suma total concurrente
        t_ini = System.currentTimeMillis();
        for(i = 0; i < n; i++){
            hilos[i].start();
        }
        try{
            for(i = 0; i < n; i++){
                hilos[i].join();
            }   
        }catch(Exception e){
            System.out.println("tirar exception con e.getmessage()");
        }
        t_fin = System.currentTimeMillis();
        t_total = t_fin - t_ini;

        // Computar la suma total de las sum parciales de cada hilo
        total = 0;
        for(i = 0; i < n; i++){
            total += hilos[i].getAcum();
        }
        System.out.println("La suma total: " + total);
        System.out.println("La suma total con " + n + " hilos tardo: " + t_total);
    }
}