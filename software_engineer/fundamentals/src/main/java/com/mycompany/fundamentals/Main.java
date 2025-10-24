/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.fundamentals;

/**
 *
 * @author joaco
 */
public class Main {
    public static void main(String[] args) {
        
        Continente[] continentes = Continente.values();
        for (int i = 0; i < continentes.length; i++) {
            Continente continente = continentes[i];
            System.out.println(continente 
                    + "\nse encuentra en la posicion: "+ continente.ordinal()
                    + "\nsu densidad es: " + continente.densidad());
            
        }

    }
    /*Motor m1 = new Motor(1000, 100, true, 4);
        Motor m2 = new Motor(4000, 30, true, 8);
        Motor m3 = new Motor(3000, 60, true, 3);
        Motor m4 = new Motor(2000, 10, true, 6);

        Auto a1 = new Auto("ABC123", "Rojo", Marca.VOKLSWAGEN, 30.45, m1);
        Auto a2 = new Auto("DEF456", "Azul", Marca.CHEVY, 60.4, m2);
        Auto a3 = new Auto("GHI654", "Negro", Marca.VOKLSWAGEN, 13.5, m3);
        Auto a4 = new Auto("JKL321", "Blanco", Marca.FIAT, 35.5, m4);

        System.out.println(a1);
        System.out.println(a2);
        System.out.println(a3);
        System.out.println(a4);
        */
}
