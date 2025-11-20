package CL2;

import TDA.Nodo;
import TDA.OperacionesCL2;
//IMPLEMENTACION DE LISTA OBJETOS CON SLL
public abstract class Lista3SLLinkedL implements OperacionesCL2{
    private Nodo FrenteL;
    private int ultimo;

    public Lista3SLLinkedL(){
        limpiar();
    }

    public void limpiar(){
        this.FrenteL=null;
        this.ultimo=-1;
    }

    public int tamanio(){
        return this.ultimo+1;
    }

    public boolean estaVacia(){
        return this.FrenteL==null;
    }
    
    public void insertar(Object elemento){
        Nodo temp;
        boolean flag;
        if(estaVacia()){
            FrenteL=new Nodo(elemento);
            this.ultimo++;
        }else{
            if(esMenor(elemento,FrenteL.getNodoInfo()))
                FrenteL=new Nodo(elemento,FrenteL);
            else{
                flag=true;
                temp=FrenteL;
                while (temp.getNextNodo()!=null&&flag) {
                    if(esMenor(elemento,temp.getNextNodo().getNodoInfo()))
                        flag=false;
                    else
                        temp=temp.getNextNodo();
                }//error cuando el mas grande de la lista es el mayor de la lsita final los siguientes no se ordenan atras
                temp.setNextNodo(new Nodo(elemento,temp.getNextNodo()));
            }
            this.ultimo++;
        }
    }

    public void eliminar(int pos){
        int c=0;
        Nodo temp=FrenteL;
        if(!estaVacia()){
            if(pos<tamanio()){
                if(pos==0){
                    FrenteL=FrenteL.getNextNodo();
                }else{
                    while (c!=pos-1) {
                        temp=temp.getNextNodo();
                        c++;
                    }
                    temp.setNextNodo(temp.getNextNodo().getNextNodo());
                }
            }else{System.out.println("Error posicion no existente");}
        }else{System.out.println("Error lista vacia");}    
    }

    public Object devolver(int pos){
        Object elem=null;
        int c=0;
        Nodo temp=FrenteL;
        if(!estaVacia()){
            if(pos<tamanio()){
                if(pos==0){
                    elem = FrenteL.getNodoInfo();
                }else{
                    while (c<pos) {
                        temp=temp.getNextNodo();
                        c++;
                    }
                    elem=temp.getNodoInfo();
                    System.err.println("LLEGUE");
                }
            }else{System.out.println("Error posicion no existente");}
        }else{System.out.println("Error lista vacia");}
        return elem;
    }

    public int buscar(Object elem){
        int c=-1;
        Nodo temp=FrenteL;
        if(!estaVacia()){
            while (!iguales(elem,temp.getNodoInfo())&&esMayor(elem,temp.getNodoInfo())&&temp!=null) {
                c++;
                temp=temp.getNextNodo();
            }
            if(temp==null||!esMayor(elem,temp.getNodoInfo()))
                c=-1;
        }else{System.out.println("Error lista vacia");}

        if(c==-1)
            System.err.println("Error no se encuentra en la lista");
        return c;
    }

    public abstract boolean esMenor(Object elem1,Object elem2);
    public abstract boolean iguales(Object elem1,Object elem2);
    public abstract boolean esMayor(Object elem1,Object elem2);
}
