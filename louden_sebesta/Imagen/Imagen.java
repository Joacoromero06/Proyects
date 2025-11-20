public class Imagen{
    int[][] matriz;
    int m, n;
    public Imagen(int m, int n){
        this.matriz = new int[m][n];
        this.m = m;
        this.n = n;
        generarImagenAleatoria();
    }
    public void generarcionAleatoria(){
        int i, j;
        for(i = 0; i < this.m; i++){
            for(j = 0; j < this.n; j++){
                setValorAleatorio(i, j);
            }
        }
    }
    public void setValor(int i, int j, int valor){
        this.matriz[i][j] = valor;
    }
    public void setValorAleatorio(int i, int j){
        java.util.Random rand = new java.util.Random();
        setValor(i, j, rand.nextInt(256));
    }
    public int getValor(int i, int j){
        return this.matriz[i][j];
    }
    public void mostrar(){
        int i, j;
        for(i = 0; i < this.m; i++){
            System.out.print("{");
            for( j = 0; j < this.n; j++){
                System.out.print(getValor(i, j));
                if( j < this.n - 1 ){
                    System.out.print(", ");
                }
            }
            System.out.print("}");
        }

    }

    public void loadImage(){
        //traer la imagen  y la convierto en blanco y negro
    }
    public int[ ] get Histograma(){
        //color 
    }
}
//file chooser