package org.example;

public class HiloMsje extends  Thread{
    String msje;
    int cant;
    Contador contador;
    Buffer buffer;

    public  HiloMsje(String msje, int cant, Contador contador, Buffer buffer){
        this.msje = msje;
        this.cant = cant;
        this.contador = contador;
        this.buffer = buffer;
    }

    public  void run(){
        int i;

        for( i = 0; i < this.cant; i++) {
            this.buffer.add(this.msje);
            this.contador.incrementar();
            System.out.println(msje + this.contador.c);
        }
    }

}
