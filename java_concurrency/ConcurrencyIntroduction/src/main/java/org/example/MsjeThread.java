package org.example;

public class MsjeThread implements  Runnable{

    @Override
    public void run() {
        int i;

        for(i = 0; i < 10; i++)
            System.out.println("inside "+ Thread.currentThread() + " " + i);
    }
}
