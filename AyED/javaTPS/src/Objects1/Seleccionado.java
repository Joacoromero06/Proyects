package Objects1;

import assets.Entrada;

public class Seleccionado {
    private String Pais;
    private int Ganados, EmpPer;
    
    public Seleccionado(String Pais, int ganados, int empPer) {
        this.Pais = Pais;
        this.Ganados = ganados;
        this.EmpPer = empPer;
    }
    public Seleccionado(){
        this(null,0,0);
    }
    public void Carga1(){
        System.err.println("Ingrese Pais: ");
        this.Pais=Entrada.sc.next();
        System.out.println("Ingrese cantidad de partidos ganados: ");
        this.Ganados=Entrada.sc.nextInt();
        System.out.println("Ingrese cantidad de partidos perdidos y empatados: ");
        this.EmpPer=Entrada.sc.nextInt();
        
    }
    public boolean esMenor(Seleccionado p1){
        if(this.esIgual(p1))
            return this.Pais.compareTo(p1.Pais)<0;
        else 
            return (this.Ganados-this.EmpPer)<(p1.Ganados-p1.EmpPer);
    }
    public boolean esMayor(Seleccionado p1){
        if(this.esIgual(p1))
           return this.Pais.compareTo(p1.Pais)>0;
        else 
            return (this.Ganados-this.EmpPer)>(p1.Ganados-p1.EmpPer);
    }
    public boolean esIgual(Seleccionado p1){
        return (this.Ganados-this.EmpPer)==(p1.Ganados-p1.EmpPer);
    }
    public String toString(){
        return "Seleccionado: "+this.Pais+" Ganados : "+this.Ganados+" PerdEmp: "+this.EmpPer;
    }

}
