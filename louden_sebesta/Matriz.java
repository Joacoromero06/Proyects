public class Matriz {
    double[][] matriz;
    int n, m;

    public Matriz(int m, int n, int ini, int fin){
        this.matriz = new double[m][n];
        this.m = m;// matriz con filas de 0 a m-1
        this.n = n;//matriz con columnas de 0 a n-1
        generarcionAleatoria(ini, fin);
    }
    public void setValor(int m, int n, double valor){
        this.matriz[m][n] = valor;
    }
    public double getValor(int m, int n){
        return this.matriz[m][n];
    }
    public void setValorAleatorio(int m, int n, int ini, int fin){
        java.util.Random rand = new java.util.Random();
        this.matriz[m][n] = rand.nextDouble()*(fin-ini) + ini;
    }
    public void generarcionAleatoria(int ini, int fin){
        int i, j;
        for(i = 0; i < this.m; i++){
            for(j = 0; j < this.n; j++){
                setValorAleatorio(i, j, ini, fin);
            }
        }
    }
    public void mostrar(){
        int i, j;
        System.out.println("{");
        for(i = 0; i < this.m; i++){
            System.out.print("{");
            for(j = 0; j < this.n; j++){
                System.out.print(getValor(i, j));
                if( j < this.n - 1)
                    System.out.print(", ");
            }
            System.out.println("}");
        }
        System.out.println("{");
    }
    public Matriz sumarSubMatriz(int fila_ini, int fila_fin, int col_ini, int col_fin, Matriz B){
        int i, j;
        Matriz C = new Matriz(this.m, this.n, 0, 0);
        for(i = fila_ini; i < fila_fin; i++){
            for( j = col_ini; j < col_fin; j++){
                double suma = getValor(i,j) + B.getValor(i,j);
                System.out.println("El valor de la suma es: " + suma );
                System.out.println("porque Aij = " + getValor(i,j) + " Bij = " + B.getValor(i,j));
                C.setValor(i, j, suma);
            }
        }
        return C;
    }
    public double sumaTotalPorSubMatriz(int fila_ini, int fila_fin, int col_ini, int col_fin){
        int i, j;
        double acum;
        acum = 0;
        for(i = fila_ini; i < fila_fin; i++){
            for( j = col_ini; j < col_fin; j++){
                acum += getValor(i,j);
                //System.out.println("El valor de la suma es: " + suma );
                //System.out.println("porque Aij = " + getValor(i,j) + " Bij = " + B.getValor(i,j));     
            }
        }
        return acum;
    }

    public double sumaTotal(){
        int i, j;
        double total = 0;
        for(i = 0; i < this.m; i++){
            for(j = 0; j < this.n; j++){
                total += getValor(i,j);
            }
        }
        return total;
    }
    public int getColumnas(){
        return this.m;
    }
    public int getFilas(){
        return this.n;
    }
}