package CL1;

import TDA.Nodo;

public abstract class ColaPrioridadSLL extends ColaSLinkedList {
    public void meter(Object elem){
        if(estaVacia()){
            this.frenteC=new Nodo(elem);
            this.finalC=this.frenteC;
        }
        else{
            if(esMenor(elem, this.frenteC.getNodoInfo())){
                this.frenteC=new Nodo(elem,this.frenteC);
            }
            else{
                if(esMayor(elem, this.finalC.getNodoInfo())){
                    this.finalC=new Nodo(elem,this.frenteC);
                }
                else{
                Nodo temp=this.frenteC;
                while (temp.getNextNodo()!=null&& (esMayor(elem, temp.getNextNodo().getNodoInfo()) || iguales(elem, temp.getNextNodo().getNodoInfo()))) 
                    temp=temp.getNextNodo();
                temp.setNextNodo(new Nodo(elem,temp.getNextNodo()));    
                }
            }
        }
    }

       
    public abstract boolean iguales(Object elemento1, Object elemento2);
	public abstract boolean esMenor(Object elemento1, Object elemento2);
	public abstract boolean esMayor(Object elemento1, Object elemento2);
}
