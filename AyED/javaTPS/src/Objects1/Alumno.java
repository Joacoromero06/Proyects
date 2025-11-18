package Objects1;

import java.util.Random;

public class Alumno {
    public int Usuario=1;
    private int TiempoNecesita,TiempoRestante;

    public Alumno(int tiempoNecesita, int tiempoRestante,int Usuario) {
        TiempoNecesita = tiempoNecesita;
        TiempoRestante = tiempoRestante;
        this.Usuario=Usuario;
    }
    public Alumno(int Usuario,int aleatorio){
        this(aleatorio,aleatorio,Usuario);
    }
    public Alumno(int Usuario){
        this(Usuario,30 + new Random().nextInt(31));
    }
    public void PasaTiempo(){
        this.TiempoRestante--;
    }
    public boolean TerminoTiempo(){
        return this.TiempoRestante==0;
    }
    public String toString(){
        return "Usuario: "+Usuario+" Tiempo necesitado: "+this.TiempoNecesita+" Tiempo Restante: "+this.TiempoRestante;
    }
    
}
