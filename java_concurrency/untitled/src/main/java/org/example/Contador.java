package org.example;

public class Contador {
    int c;

    public Contador(int c) {
        this.c = c;
    }
    public int getC(){
        return  this.c;
    }
    public  void setC(int c){
        this.c = c;
    }

    public  void incrementar(){
        this.c = this.c +1;
    }
    public  void show(){
        System.out.println(this.c);
    }
}
