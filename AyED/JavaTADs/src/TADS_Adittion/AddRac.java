package TADS_Adittion;

import Recursos.IOpeAvanzadas;

import TADs.Racional;

public class AddRac extends Racional implements IOpeAvanzadas{
    public void Division(Object a,Object b){
        
        Racional aux=new Racional(((Racional)b).getDen(), ((Racional)b).getNum());
        Racional div=new Racional();
        
        div.Producto(aux, a);//ya simplifica
        
        setNum(div.getNum());//el obj que llamo el metodo guarda la division
        setDen(div.getDen());//resuelto si era private 
    }
}
