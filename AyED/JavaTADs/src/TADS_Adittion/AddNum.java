package TADS_Adittion;

import Recursos.IOpeAvanzadas;

import Recursos.ParOrdenado;

import TADs.Numero;

public class AddNum extends Numero implements IOpeAvanzadas{

    public void Division(Object a, Object b) {
        ParOrdenado cyr=Numero.CyR((Numero)a,(Numero)b);
        SetValor(cyr.getX().GetValor());
    }
    
}
