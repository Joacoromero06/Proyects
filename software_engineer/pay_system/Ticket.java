package pay_system;

public class Ticket {
    Persona _comprador;
    double _monto_total;
    double _monto_x_cuota;


    public Ticket(Persona _comprador, double _monto_total, double _monto_x_cuota) {
        this._comprador = _comprador;
        this._monto_total = _monto_total;
        this._monto_x_cuota = _monto_x_cuota;
    }

    public Persona get_comprador() {
        return this._comprador;
    }

    public void set_comprador(Persona _comprador) {
        this._comprador = _comprador;
    }

    public double get_monto_total() {
        return this._monto_total;
    }

    public void set_monto_total(double _monto_total) {
        this._monto_total = _monto_total;
    }

    public double get_monto_x_cuota() {
        return this._monto_x_cuota;
    }

    public void set_monto_x_cuota(double _monto_x_cuota) {
        this._monto_x_cuota = _monto_x_cuota;
    }

    @Override
    public String toString() {
        return "{" +
            " _comprador='" + get_comprador() + "'" +
            ", _monto_total='" + get_monto_total() + "'" +
            ", _monto_x_cuota='" + get_monto_x_cuota() + "'" +
            "}";
    }
}