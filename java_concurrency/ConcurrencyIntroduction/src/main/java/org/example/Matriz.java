package org.example;

public class Matriz {
    double[][] matriz;
    int n, m;

    public Matriz(int m, int n){
        this.matriz = new double[m][n];
        this.m = m;
        this.n = n;
        carga();
    }
    public void carga(){
        int i, j;
        for( i = 0; i < this.m; i++){
            for (j = 0; j < this.n; j++){
                this.matriz[i][j] = 0;
            }
        }
    }
    public void muestra(){
        int i, j;
        System.out.println("{");
        for( i = 0; i < this.m; i++){
            System.out.print("{");
            for (j = 0; j < this.n; j++){
                System.out.print(this.matriz[i][j]);
                if (j + 1 < this.n){
                    System.out.print(",");
                }
            }
            System.out.print("}");
            if (i + 1 < this.m){
                System.out.println(",");
            }
        }
        System.out.println("}");

    }
}
