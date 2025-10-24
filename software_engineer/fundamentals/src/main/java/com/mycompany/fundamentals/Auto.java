/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.fundamentals;

/**
 *
 * @author joaco
 */
public class Auto {
    // Atributos de instancia
    private String patente;
    private String color;
    private Marca marca;
    private double precio;
    private Motor motor;

    // Atributos de clase
    private static double DESCUENTO;


    public Auto(String patente, String color, Marca marca, double precio, Motor motor) {
        this.patente = patente;
        this.color = color;
        this.marca = marca;
        this.precio = precio;
        this.motor = motor;
    }

    public double precio_promocional() {
        return this.precio * Auto.DESCUENTO;
    }

    @Override
    public String toString() {
        return "{" +
            " patente='" + getPatente() + "'" +
            ", color='" + getColor() + "'" +
            ", marca='" + getMarca() + "'" +
            ", precio='" + getPrecio() + "'" +
            ", motor='" + getMotor() + "'" +
            "}";
    }

    public String getPatente() {
        return this.patente;
    }

    public void setPatente(String patente) {
        this.patente = patente;
    }

    public String getColor() {
        return this.color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public Marca getMarca() {
        return this.marca;
    }

    public void setMarca(Marca marca) {
        this.marca = marca;
    }

    public double getPrecio() {
        return this.precio;
    }

    public void setPrecio(double precio) {
        this.precio = precio;
    }

    public Motor getMotor() {
        return this.motor;
    }

    public void setMotor(Motor motor) {
        this.motor = motor;
    }

    
}