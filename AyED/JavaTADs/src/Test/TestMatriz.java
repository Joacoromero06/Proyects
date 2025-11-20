package Test;

import TADs.Matriz;

public class TestMatriz {
    public static void main(String[] args) {
        Matriz aux=new Matriz();

        Matriz A= (Matriz)aux.Carga();
        System.out.println(A);

        Matriz B= (Matriz)aux.Carga();
        System.out.println(B);

        Matriz C= (Matriz)aux.Inicializa();
        C.Sumar(A, B);
        System.out.println(C);

        C.Potencia(C, 2);
        System.out.println(C);

    }
     
}