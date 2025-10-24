package pay_system;

import java.util.concurrent.ThreadLocalRandom;
public class Main { 
    
    public static void main(String[] args){
        Posnet posnet = new Posnet();
        TarjetaCredito tarjeta = new TarjetaCredito(15000.0);
        posnet.efectuar_pago(tarjeta, 10000, 6);
        System.out.println(tarjeta);


    }
}