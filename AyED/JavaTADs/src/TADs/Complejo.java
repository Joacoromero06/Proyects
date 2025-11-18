package TADs;

import Recursos.Entrada;

import Recursos.Operable;

public class Complejo implements Operable{
    double real,imaginaria;

    //CONSTRUCTORES SETTERS Y GETTERS


    public Complejo(){
        this(0,0);
    }
    public Complejo(double real, double imaginaria) {
        this.real = real;
        this.imaginaria = imaginaria;
    }
    public double getReal() {
        return real;
    }
    public void setReal(double real) {
        this.real = real;
    }
    public double getImaginaria() {
        return imaginaria;
    }
    public void setImaginaria(double imaginaria) {
        this.imaginaria = imaginaria;
    }

    
    //METODOS DE OPERABLE GENERALES PARA CUALQUIER OBJETO MATEMATICO
    public void Sumar(Operable a, Operable b) {
        this.real=((Complejo)a).getReal()+((Complejo)b).getReal();
        this.imaginaria=((Complejo)a).getImaginaria()+((Complejo)b).getImaginaria();
    }

    
    public void Resta(Operable a, Operable b) {
        this.real=((Complejo)a).getReal()-((Complejo)b).getReal();
        this.imaginaria=((Complejo)a).getImaginaria()-((Complejo)b).getImaginaria();
    }

    
    public void Producto(Operable a, Operable b) {
        this.real=((Complejo)a).getReal()*((Complejo)b).getReal();
        this.imaginaria=((Complejo)a).getImaginaria()*((Complejo)b).getImaginaria();
    }

   
    public void Potencia(Operable a, int n) {
        this.real=Math.pow(((Complejo)a).getReal(),n);
        this.imaginaria=Math.pow(((Complejo)a).getImaginaria(),n);
    }

    
    public boolean Iguales(Operable a, Operable b) {
        return ((Complejo)a).getReal()==((Complejo)b).getReal() &&((Complejo)a).getImaginaria()==((Complejo)b).getImaginaria();
    }


    public Operable Carga(){
        System.out.println("Carga Complejo");

        System.out.println("Ingrese la parte real: ");
        double real=Entrada.sc.nextDouble();
        
        System.out.println("Ingrese la parte real: ");
        double imaginaria=Entrada.sc.nextDouble();

        return new Complejo(real,imaginaria);
    }
    
    
    public Operable Inicializa(){
        return new Complejo(); 
    }
    
    
    //METODOS UNICOS DE LA CLASE COMPLEJO
    public String toString(){
        return "Numero Complejo: "+this.real+" "+this.imaginaria+"i ";
    }

    
    public void Conjugado(Complejo a){
        this.real=a.getReal();
        this.imaginaria=a.getImaginaria()*-1;
    }


    /*public void Division(Complejo a, Complejo b){
        Complejo aux=new Complejo();
        if(!Iguales(aux, b)){
            this.real=a.getReal()/b.getReal();
            this.imaginaria=a.getImaginaria()/b.getImaginaria();
        }
        else
            System.err.println("Division por 0+0i no definida");
    }*/


    public static double Modulo(Complejo a){
        return Math.sqrt(a.getReal()*a.getReal()+a.getImaginaria()*a.getImaginaria());
    }
}
