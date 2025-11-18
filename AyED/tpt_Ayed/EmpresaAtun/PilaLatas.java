package EmpresaAtun;

import CL1.PilaArr;
import CL1.PilaSLinkedList;

public class PilaLatas extends PilaArr{
    public PilaLatas (int tam){
        super(tam);
    } 
    public Lata sacar (){
        return ((Lata)super.sacar());
    } 

    public String toString(){
        PilaSLinkedList aux=new PilaSLinkedList();
        int c=0;
        String str="\n";
        Lata lata;
        while (!this.estaVacia()) {
            lata=this.sacar();
            aux.meter(lata);
            c++;
            str+=lata+"\n";
        }
        while (!aux.estaVacia()) {
            super.meter((Lata)aux.sacar());
        }
        str+=" La pila tiene "+c+" elementos";
        return str;
    }
    
    public int tamanio(){
        PilaSLinkedList aux=new PilaSLinkedList();
        int c=0; 
        Lata lata;
        while (!this.estaVacia()) {
            lata=this.sacar();
            aux.meter(lata);
            c++; 
        }
        while (!aux.estaVacia()) {
            super.meter(aux.sacar());
        }
        return c;
    }
}