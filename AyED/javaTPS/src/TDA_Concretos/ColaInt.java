package TDA_Concretos;

import assets.Entrada;
import TDA.ColaUtil;

public class ColaInt extends ColaUtil{
    

    public void Carga(){
        System.out.println("Vamos a cargar la Cola: ");
        System.out.println("Ingrese el tamaño: ");
        int n=Entrada.sc.nextInt();

        for(int i=0;i<n;i++){
            System.out.println("Cargue el elemento "+(i+1)+": ");
            this.cola.meter(Entrada.sc.nextInt());
        }
    }


    public String toString() {

        String str="Vamos a mostrar la cola: \n";
        ColaInt aux=new ColaInt();
        int elem;

        while(!this.cola.estaVacia()){
            str+="Elemento: ";
            elem=(int)this.cola.sacar();
            str+=elem;
            aux.cola.meter(elem);
            str+="\n";
        }

        while (!aux.cola.estaVacia()) {
            this.cola.meter(aux.cola.sacar());
        }
        return str;
    }



}
