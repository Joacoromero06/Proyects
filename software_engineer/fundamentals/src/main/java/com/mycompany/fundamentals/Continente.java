/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.fundamentals;

/**
 *
 * @author joaco
 */
public enum Continente {
    AMERICA(50000, 500l),
    ASIA(70000, 700l),
    AFRICA(60000, 600l),
    EUROPA(30000, 300l),
    OCEANIA(20000, 20000000l);
    private int habitantes;
    private long superficie;
    
    private Continente(int habitantes, long superficie){
        this.habitantes = habitantes;
        this.superficie = superficie;
    }
    public double densidad(){
        return habitantes/superficie;
    }

    @Override
    public String toString() {
        return "Continentes{" + "ordinal=" + ordinal() + ", name=" + name() + ", habitantes=" + habitantes + ", superficie=" + superficie + '}';
    }
    
}
