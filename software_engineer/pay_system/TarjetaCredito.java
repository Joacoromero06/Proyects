package pay_system;

import java.util.concurrent.ThreadLocalRandom;

public class TarjetaCredito{
    private Persona _titular;
    private String _entidad_bancaria;
    private long _numero;
    private double _saldo;

    public TarjetaCredito(Persona _titular, String _entidad_bancaria, long _numero, double _saldo) {
        this._titular = _titular;
        this._entidad_bancaria = _entidad_bancaria;
        this._numero = _numero;
        this._saldo = _saldo;
    }
    public TarjetaCredito(double saldo){
        this._titular = new Persona();
        
        ThreadLocalRandom rand = ThreadLocalRandom.current();
        String[] entidades_bancarias = {"Santander", "ICBC", "Mercado Pago", "Personal Pay"};
        this._entidad_bancaria = entidades_bancarias[rand.nextInt(entidades_bancarias.length)];
        
        this._numero = rand.nextLong(10000000, 100000000);
        this._saldo = saldo;
    }
    public boolean _saldo_suficiente(double monto){
        boolean b;
        if(this._saldo >= monto)
            b = true;
        else
            b = false;
        return b;
    }
    public void _pagar(double monto){
        if (_saldo_suficiente(monto))
            this._saldo = this._saldo - monto;
        
    }

    public Persona get_titular() {
        return this._titular;
    }

    public void set_titular(Persona _titular) {
        this._titular = _titular;
    }
  
    public String get_entidad_bancaria() {
        return this._entidad_bancaria;
    }

    public void set_entidad_bancaria(String _entidad_bancaria) {
        this._entidad_bancaria = _entidad_bancaria;
    }

    public long get_numero() {
        return this._numero;
    }

    public void set_numero(long _numero) {
        this._numero = _numero;
    }

    public double get_saldo() {
        return this._saldo;
    }

    public void set_saldo(double _saldo) {
        this._saldo = _saldo;
    }


    @Override
    public String toString() {
        return "{" +
            " _titular='" + get_titular() + "'" +
            ", _entidad_bancaria='" + get_entidad_bancaria() + "'" +
            ", _numero='" + get_numero() + "'" +
            ", _saldo='" + get_saldo() + "'" +
            "}";
    }
    


}
