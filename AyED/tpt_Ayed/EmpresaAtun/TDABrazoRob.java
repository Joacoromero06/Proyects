package EmpresaAtun;

public interface TDABrazoRob {
    public void Procesar1Lata(Lata lata);
    public void Procesar(Object procesado);
    public void Detener(int razon);
    public void MuestraSituacionBrazo();
    public void GeneraInforme();
    public void Aviso(int razon);
}
