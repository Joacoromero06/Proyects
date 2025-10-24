package pay_system;

import java.util.concurrent.ThreadLocalRandom;

public class Persona{
    private long _DNI;
    private long _telefono;
    private String _nombre;
    private String _apellido;
    private String _mail;

    public Persona(){
        ThreadLocalRandom rand = ThreadLocalRandom.current();
        
        this._DNI = rand.nextLong(10000000, 60000000);
        this._telefono = rand.nextLong(10000, 100000);

        String[] nombres = {"Juan","Joaco","Juli","Julian","Joma"};
        String[] apellidos = {"Romero","Reyes","Pareja","Trenti","Masse"};       

        this._nombre = nombres[rand.nextInt(nombres.length)];
        this._apellido = apellidos[rand.nextInt(apellidos.length)];
        
        this._mail = this._nombre + '.' + this._apellido + "@gmail.com";
    }


    public long get_DNI() {
        return this._DNI;
    }

    public void set_DNI(long _DNI) {
        this._DNI = _DNI;
    }

    public long get_telefono() {
        return this._telefono;
    }

    public void set_telefono(long _telefono) {
        this._telefono = _telefono;
    }

    public String get_nombre() {
        return this._nombre;
    }

    public void set_nombre(String _nombre) {
        this._nombre = _nombre;
    }

    public String get_apellido() {
        return this._apellido;
    }

    public void set_apellido(String _apellido) {
        this._apellido = _apellido;
    }

    public String get_mail() {
        return this._mail;
    }

    public void set_mail(String _mail) {
        this._mail = _mail;
    }


    @Override
    public String toString() {
        return "{" +
            " _DNI='" + get_DNI() + "'" +
            ", _telefono='" + get_telefono() + "'" +
            ", _nombre='" + get_nombre() + "'" +
            ", _apellido='" + get_apellido() + "'" +
            ", _mail='" + get_mail() + "'" +
            "}";
    }
    
    
}
