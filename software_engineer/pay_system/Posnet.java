package pay_system;

public class Posnet {
    public static final double _RECARGO_POR_CUOTAS = 0.03;
    public static final int _CANTIDAD_MIN_CUOTAS = 1;
    public static final int _CANTIDAD_MAX_CUOTAS = 6;

    public Ticket efectuar_pago(TarjetaCredito tarjeta, double monto_abonar, int cant_cuotas){
        Ticket ticket = null;
        double monto_mas_regargo = 0.0;
        double monto_por_cuota = 0.0;
        
        if ( valida_datos( tarjeta, monto_abonar, cant_cuotas ) ){
            monto_mas_regargo = aplica_regargo(monto_abonar, cant_cuotas);
            
            if( tarjeta._saldo_suficiente(monto_mas_regargo) ){
                System.out.println("pagando..." + monto_mas_regargo);
                tarjeta._pagar(monto_mas_regargo);
                monto_por_cuota =  monto_mas_regargo / cant_cuotas;
                ticket = new Ticket(tarjeta.get_titular(),monto_mas_regargo, monto_por_cuota);
            } 
        }
        return ticket;

    }

    public boolean valida_datos(TarjetaCredito tarjeta, double monto_abonar, int cant_cuotas){
        boolean b;
        if (tarjeta != null && monto_abonar > 0 && valida_cuotas(cant_cuotas)){
            b = true;
        }   
        else{
            System.out.println("Datos invalidos");
            b = false;
        }
        return b;
    }

    public boolean valida_cuotas(int cuotas){
        boolean b;
        if (cuotas <= _CANTIDAD_MAX_CUOTAS && cuotas >= _CANTIDAD_MIN_CUOTAS){
            b = true;
        }   
        else{
            System.out.println("Datos invalidos");
            b = false;
        }
        return b; 
        
    }

    public double aplica_regargo(double monto, int cant_cuotas){
        double monto_mas_regargo = 0;
        if(valida_cuotas(cant_cuotas)){
            monto_mas_regargo = monto + (cant_cuotas - 1) * _RECARGO_POR_CUOTAS * monto;
        }
        return monto_mas_regargo;
        
    }

}