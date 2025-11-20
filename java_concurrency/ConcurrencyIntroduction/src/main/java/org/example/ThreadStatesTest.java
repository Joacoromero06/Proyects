package org.example;

public class ThreadStatesTest {
    public static void main(String[] args){
        Thread t1 = new Thread( () -> {
            try {
                Thread.sleep(1);
                for(int i = 0; i < 9999; i++);
            }catch (Exception e){ e.printStackTrace();}
        }, "Thread for show states");
        t1.start();
        while (true){
            Thread.State s = t1.getState();
            System.out.println(s);
            if(s == Thread.State.TERMINATED) break;

        }
    }

}
