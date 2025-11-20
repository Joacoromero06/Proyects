import java.util.Random;

public class Imagen{
    int[][] matriz;
    int filas;
    int columnas;

    public Imagen(int m, int n){ // m filas, n columnas
        this.matriz = new int[m][n];
        filas = m;
        columnas = n;
        generacionImagen();
    }
    public void generacionImagen(){
        int i, j;
        Random rand = new Random();
        for( i = 0; i < filas; i++){
            for( j = 0; j < columnas; j++){
                matriz[i][j] = rand.nextInt(256);
            }
        }
    }
    public int getMayor(int fila_ini, int fila_fin, int col_ini, int col_fin){
        int mayor = -1;
        int i, j;
        for(i = fila_ini; i < fila_fin; i++)
            for( j = col_ini; j < col_fin; j++)
                if (matriz[i][j] > mayor)
                    mayor = matriz[i][j];
        
        if(mayor == -1) System.out.println("Error mayor es -1");
        return mayor;
    }

    public void mostrarImagen(){
        int i, j;
        System.out.println();
        for( i = 0; i < this.filas; i++){
            System.out.print("[ ");
            for( j = 0; j < this.columnas; j++){
                if(j+1 != columnas) System.out.print(matriz[i][j]+" , ");
                else System.out.print(matriz[i][j]);
            }
            System.out.print(" ]");
            System.out.println();
        }
        System.out.println();

    }

    public int getFilas(){
        return this.filas;
    }
    
    public int getColumnas(){
        return this.columnas;
    }
    public int devolver( int i, int j){
        return this.matriz[i][j];
    }
    public void actualizar(int valor, int i, int j){
        this.matriz[i][j] = valor;
    }
} 
/* 
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class Image {

    private BufferedImage imagen;

    public Image() {
        loadImage();
    }

    public void loadImage() {
        JFileChooser chooser = new JFileChooser();
        int result = chooser.showOpenDialog(null);

        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = chooser.getSelectedFile();
            try {
                BufferedImage originalImage = ImageIO.read(selectedFile);
                this.imagen = new BufferedImage(
                    originalImage.getWidth(),
                    originalImage.getHeight(),
                    BufferedImage.TYPE_BYTE_GRAY
                );
                
                Graphics g = this.imagen.getGraphics();
                g.drawImage(originalImage, 0, 0, null);
                g.dispose();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void saltAndPepper(float r) {
        // Para implementar
    }

    public void displayImage() {
        if (this.imagen != null) {
            JFrame frame = new JFrame("Imagen en escala de grises");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(this.imagen.getWidth(), this.imagen.getHeight());
            frame.add(new JLabel(new ImageIcon(this.imagen)));
            frame.pack();
            frame.setVisible(true);
        } else {
            System.out.println("No se ha cargado ninguna imagen.");
        }
    }
}
 */