package TDA;

public interface Conjunto {
    public void limpiar();
    public boolean estaVacia();
    public boolean pertenece(Object elemento);
    public void meter(Object elemento);
    public void sacar(Object elemento); 
}
