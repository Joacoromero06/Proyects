public class Main{
    public static void main(String[] args){
        int m, n;
        m = 5;
        n = 5;
        Imagen img = new Imagen(m, n);
        Imagen imgRto = new Imagen((img.getFilas()+1)/2, (img.getColumnas()+1)/2);

        img.mostrarImagen();
        //Thread hilos []=new Thread();    
        Hilos h1 = new Hilos(img,0,n/2, imgRto);
        Hilos h2 = new Hilos(img,n/2,n, imgRto);

        h1.start();
        h2.start();

        try{
            h1.join();
            h2.join();
        }catch (Exception e) { }//creo que la primera I de interrupted era mayus pero me dio error

        h1.getRdo().mostrarImagen(); 
             
╰λ java Main

[ 137 , 52 , 86 , 99 , 63 ]
[ 164 , 3 , 221 , 2 , 60 ]
[ 137 , 235 , 17 , 128 , 132 ]
[ 227 , 27 , 80 , 70 , 19 ]
[ 42 , 86 , 246 , 171 , 206 ]

soy un hilo, actualizo: 164en posicion: 0 0
soy un hilo, actualizo: 235en posicion: 1 0
soy un hilo, actualizo: 86en posicion: 2 0
soy un hilo, actualizo: 221en posicion: 0 1
soy un hilo, actualizo: 63en posicion: 0 2
soy un hilo, actualizo: 128en posicion: 1 1
soy un hilo, actualizo: 132en posicion: 1 2
soy un hilo, actualizo: 246en posicion: 2 1
soy un hilo, actualizo: 206en posicion: 2 2

[ 164 , 221 , 63 ]
[ 235 , 128 , 132 ]
[ 86 , 246 , 206 ]


    
    }
    
}