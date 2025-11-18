package TDA;

import CL1.*;

public class ColaUtil {
    protected OperacionesCL1 cola;


    public ColaUtil(){
        this.cola=new ColaSLinkedList();
    }


    public ColaUtil(int tam){
        this.cola=new ColaArr(tam);
    }    


    public OperacionesCL1 GetCola(){
        return this.cola;
    }


    public int Cardinal(){
        int c=0;
        ColaSLinkedList aux=new ColaSLinkedList();

        while (!this.cola.estaVacia()) {
            aux.meter(this.cola.sacar());
            c++;
        }

        while (!aux.estaVacia()) {
            this.cola.meter(aux.sacar());
        }

        System.out.println("Cardinal de la cola es: ");
        return c;
    }


    public void Invertir(){
        PilaSLinkedList aux=new PilaSLinkedList();
        
        System.out.println("Invirtiendo la Cola: ");
        
        while (!this.cola.estaVacia()) {
            aux.meter(this.cola.sacar());
        }

        while (!aux.estaVacia()) {
            this.cola.meter(aux.sacar());
        }
    }


    public void Concatenar(ColaUtil nvo){
        Object elem;
        ColaUtil aux= new ColaUtil();
        System.out.println("Concatenando la Cola: ");
        
        while (!nvo.GetCola().estaVacia()) {
            elem=new Object();
            elem=nvo.GetCola().sacar();
            this.cola.meter(elem);
            aux.GetCola().meter(elem);
        }
        while (!aux.GetCola().estaVacia()) {
            nvo.GetCola().meter(aux.GetCola().sacar());
        }
    }



}
