package EmpresaAtun;

public abstract class BrazoRobotico implements TDABrazoRob{
    PilaLatas PilaA, PilaB;
    boolean stop;
    InformeBrazo informe;

    public BrazoRobotico(PilaLatas A, PilaLatas B){
        this.PilaA=A; this.PilaB=B;
        this.stop=false;
        GeneraInforme();
    }
    public void MuestraSituacionBrazo(){
        GeneraInforme();
        System.out.println(this.informe);
        System.out.println("Stack 1: "+ this.PilaA);
        System.out.println("Stack 2: "+ this.PilaB);
    } 

    public void GeneraInforme() {setInforme(new InformeBrazo(getPilaA().tamanio(),getPilaB().tamanio(),getPilaA().tamanio()+ getPilaB().tamanio() ) );} 
    public boolean Pare(){return this.stop;}
    public void Parar(){this.stop=true;}
    public void Encender(){this.stop=false;}
    public PilaLatas getPilaA() {return PilaA;}
    public void setPilaA(PilaLatas pilaA) {PilaA = pilaA;}
    public PilaLatas getPilaB() {return PilaB;}
    public void setPilaB(PilaLatas pilaB) {    PilaB = pilaB;}
    public InformeBrazo getInforme() {return informe;}
    public void setInforme(InformeBrazo informe) {this.informe = informe;}

}