package org.example;

public class ThreadTester {
    public static void main(String[] args){
        System.out.println("Inside Main");
        Thread t = new Thread( () ->
        {
            int i;
            for(i = 0; i < 10 ; i++){
                System.out.println("inside " + Thread.currentThread() + " " + i );
            }
        }, "Thread with lambdas" );
        t.start();

        System.out.println("Exit Main");

    }
}
