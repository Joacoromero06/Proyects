package GestionAtun;

import EmpresaAtun.BrazoRobotico; 
import EmpresaAtun.Lata;
import EmpresaAtun.PilaLatas;

public class BrazoGestion extends BrazoRobotico {

    
    public BrazoGestion(PilaLatas A, PilaLatas B) {
        super(A, B);
    }

    public void Procesar(Object procesado) { 
        Encender();
        CargaStock(((ColaLatas)procesado));
    }
    public void CargaStock(ColaLatas cinta){
        while (!Pare()) {
            Procesar1Lata(cinta.sacar());
            if(cinta.estaVacia())
                Detener(0);
        }
    }
    public void Procesar1Lata(Lata lata){ 
        if(lata.getEsNatural()){
            if(!getPilaA().estaLlena())
                getPilaA().meter(lata);
            else
                Detener(1);
        }
        else{
            if(!getPilaB().estaLlena())
                getPilaB().meter(lata);
            else
                Detener(-1);
        }
    }
 
    public void Detener(int razon) {
        System.err.println("Brazo de Gestion Detenido");
        if(razon==0) 
            Aviso(0);
        if(razon==1)
            Aviso(1);
        if(razon==-1)
            Aviso(-1);
        Parar();
    } 
    public void Aviso(int razon) {
        if(razon==0)
            System.err.println("Cinta de Latas Vacia, Procesada con EXITO");
        if(razon==1)
            System.err.println("Stack1 (Latas al natural) LLENO");
        if(razon==-1)
            System.err.println("Stack2 (Latas al aceite) LLENO");
    }
 
 
    
}
