package TDA;

//CONTRATO COMUN ENTRE LISTAS ORDENADOS Y COMUNES DE OBJETOS
public interface OperacionesCL2 {
	public int buscar(Object elemento);
	public Object devolver(int posicion);
	public void eliminar(int posicion);
	public void limpiar();
	public boolean estaVacia();		
	public int tamanio();
}



