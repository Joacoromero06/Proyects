package TDA_Concretos;

import assets.*;
import TDA.PilaUtil;

public class PilaInt extends PilaUtil{


    public void Carga(){
        System.out.println("Vamos a cargar la Pila: ");
        System.out.println("Ingrese el tamaño");
        int n=Entrada.sc.nextInt();

        for(int i=0;i<n;i++){
            System.out.println("Cargue el elemento "+(i+1)+": ");
            this.pila.meter(Entrada.sc.nextInt());
        }
    }


    public String toString() {

        String str="Vamos a mostrar la pila: \n";
        PilaInt aux=new PilaInt();
        int elem;

        while(!this.pila.estaVacia()){
            str+="Elemento: ";
            elem=(int)this.pila.sacar();
            str+=elem;
            aux.pila.meter(elem);
            str+="\n";
        }

        while (!aux.pila.estaVacia()) {
            this.pila.meter(aux.pila.sacar());
        }
        return str;
    }
    
    
    public void Reemplaza(int bus,int nvo){
        int elem;//pilaInt aux no importa si atras tiene array o slinkedlist
        PilaInt aux=new PilaInt();//va a recibir la pila con los elem reemplazados, despues lo vuelvo a la pil original

        System.out.println("Reemplazando los: "+bus+" por: "+nvo);
        while(!this.pila.estaVacia()){
            elem=(int)this.pila.sacar();
            if(elem==bus)
                elem=nvo;
            aux.pila.meter(elem); 
        }
        while (!aux.pila.estaVacia()) {
            this.pila.meter((int)aux.pila.sacar()); 
        }
    }


    public static boolean Verifica(String str){
        int i=0;
        boolean b=true;
        PilaInt aux=new PilaInt();

        while (i<str.length() && b) {

            if(str.charAt(i)=='(')
                aux.pila.meter(0);  

            if( str.charAt(i)==')' ){
                if(aux.pila.estaVacia())
                    b=false;
                else
                    aux.pila.sacar();
            }
            i++;
        }

        if(b==true&&!aux.pila.estaVacia())
            b=false;

        System.out.println("Vamos a verificar si la cadena: "+str+" esta balanceada: ");
        return b;
    }


    public ColaInt GeneraModifica(){
        ColaInt cola=new ColaInt();
        PilaInt aux=new PilaInt();

        int elem;
        while (!this.pila.estaVacia()) {
            elem=(int)this.pila.sacar();
            if(PropNum.esCapicua(elem))
                cola.GetCola().meter(elem);
            else 
                aux.pila.meter(elem);
        }

        while (!aux.pila.estaVacia()) {
            this.pila.meter(aux.pila.sacar());
        }

        return cola;
    }
}
