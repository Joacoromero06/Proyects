/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.fundamentals;

/**
 *
 * @author joaco
 */ 

public class Motor {
    private int caballos;
    private int cilindrada;
    private boolean esDiesel;
    private int tamanio;


    public Motor(int caballos, int cilindrada, boolean esDiesel, int tamanio) {
        this.caballos = caballos;
        this.cilindrada = cilindrada;
        this.esDiesel = esDiesel;
        this.tamanio = tamanio;
    }


    public int getCaballos() {
        return this.caballos;
    }

    public void setCaballos(int caballos) {
        this.caballos = caballos;
    }

    public int getCilindrada() {
        return this.cilindrada;
    }

    public void setCilindrada(int cilindrada) {
        this.cilindrada = cilindrada;
    }

    public boolean isEsDiesel() {
        return this.esDiesel;
    }

    public boolean getEsDiesel() {
        return this.esDiesel;
    }

    public void setEsDiesel(boolean esDiesel) {
        this.esDiesel = esDiesel;
    }

    public int getTamanio() {
        return this.tamanio;
    }

    public void setTamanio(int tamanio) {
        this.tamanio = tamanio;
    }

    
}