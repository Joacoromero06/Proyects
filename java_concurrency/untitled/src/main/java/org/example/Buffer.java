package org.example;

public class Buffer {
    String[] buffer;
    int tam;
    public  Buffer(){
        this.buffer = new String[300];
        this.tam = 0;
    }

    public  void add(String msje){
        this.buffer[this.tam] = msje;
        this.tam ++;
    }

    public void show() {
         int i;
         for( i = 0; i<this.tam; i++) System.out.println(this.buffer[i] + i);
    }
}
