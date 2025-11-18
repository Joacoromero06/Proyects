package Objects1;

public class Pc {
    private boolean ocupada;
    private Alumno alumno;
    private int nro,horario;

    public Pc(boolean ocupada,Alumno alumno,int nro,int horario) {
        this.ocupada = ocupada;
        this.alumno=alumno;
        this.nro=nro;
        this.horario=horario;
    }
    
    public Pc(int nro){
        this(false,null,nro,0);
    }
    public boolean estadoPc(boolean b){
        return this.ocupada==b;
    }

    public boolean estaOcupada(){
        return this.ocupada==true;
    }

    public void usar(Alumno alumno,int tiempoLlegada){
        this.ocupada=true;
        this.alumno=alumno;
        this.horario=tiempoLlegada;
    }
    public void desocupar(){
        this.ocupada=false;
        this.alumno=null;
    }
    public void PasaMin(){
        if(estaOcupada()){
            this.alumno.PasaTiempo();
            if(this.alumno.TerminoTiempo()){
                System.out.println("Se libera pc nro:"+this.nro+" se va alumno: "+this.alumno+" horario llegada: "+this.horario);
                desocupar();
            }
        }
        
    }

    
}
