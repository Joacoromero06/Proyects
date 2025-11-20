package EmpresaAtun;

public class InformeBrazo {
    int cantPilaA, cantPilaB;
    int StockTotal;
    public InformeBrazo(int cantPilaA, int cantPilaB, int stockTotal) {
        this.cantPilaA = cantPilaA;
        this.cantPilaB = cantPilaB;
        StockTotal = stockTotal;
    }
    
    public String toString(){return "Informe del brazo: "+ "Cantidad de latas en Stack1 (al Natural): "+ this.cantPilaA+" Cantidad de latas en Stack2 (al Aceite): "+ this.cantPilaB+" Stock Total: "+this.StockTotal;}
}
