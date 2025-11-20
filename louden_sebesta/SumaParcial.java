public class SumaParcial extends Thread{
    Matriz matriz;
    int col_ini, col_fin;
    double acum;

    public SumaParcial(Matriz m, int col_ini, int col_fin){
        this.matriz = m;
        this.col_ini = col_ini;
        this.col_fin = col_fin; // col fin no incluido
        this.acum = 0;
    }
    public void run(){
        this.acum = this.matriz.sumaTotalPorSubMatriz(0, this.matriz.getFilas(), col_ini, col_fin);
    }
    //tambien puedo usar sumarSubMatriz()
    public double getAcum(){
        return this.acum;
    }
}