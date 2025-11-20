package org.example;

public class Singleton {
    private static volatile  Singleton instance = null;
    private Singleton(){
        System.out.println("Singleton creado");
    }
    public static Singleton getSingletonInstance(){
        if(instance == null)
            synchronized (Singleton.class){
                if(instance == null) instance = new Singleton();
            }
        return instance;
    }
}
