/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.fundamentals;

import java.util.Scanner;

/**
 *
 * @author joaco
 */
public class TestErrors {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println(factorial(4));

        try{
            System.out.println("Calcularemos p! / q.");

            System.out.println("Ingrese p:");
            int p = sc.nextInt();
            System.out.println("Ingrese q:");
            int q = sc.nextInt();
            
            int rtdo = factorial(p) / q;
            System.out.println("El resultado es: " + rtdo);
        }
        catch(Exception ex){
            System.out.println("Error: " + ex.getMessage());
            ex.printStackTrace(System.out);
        }
        System.out.println("Adiós. ");
    }
    public static int factorial(int x){
        if (x < 0){
            throw new RuntimeException("No es posible calcular el factorial de un numero negativo.");
        }
        if (x > 17){
            throw new RuntimeException("Por representacion interna no se puede calcular.");
        }
        if (x == 1 || x == 0){
            return 1;
        }
        return x * TestErrors.factorial(x-1);
    }
}
