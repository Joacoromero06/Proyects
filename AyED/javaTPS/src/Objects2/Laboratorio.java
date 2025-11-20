package Objects2;

import CL1.ColaSLinkedList;
import Objects1.Alumno;

import java.util.Random;

public class Laboratorio extends ListaPc{
    private ColaSLinkedList ColaAlumnos;

    public Laboratorio(ColaSLinkedList colaAlumnos) {
        ColaAlumnos = colaAlumnos;
    }
    public Laboratorio(){
        this(new ColaSLinkedList());
    }

    public void SimulaDia(){
        InicilizaLab();
        int cantUsuarios=0;
        Random rand = new Random();
        for(int min=0;min<=599;min++){
            procesColaAlumno(min+1);
            if(rand.nextInt(100) + 1<=15){
                procesoNuevoAlumno(min+1,cantUsuarios++);
            }
            PasaMin();
        }
    }
    public void procesColaAlumno(int tiempoAct){
        int nroPc;
        if(!this.ColaAlumnos.estaVacia()){
            nroPc=BuscaPcLibre();
                if(nroPc!=-1){
                    OcuparPc(nroPc, (Alumno)this.ColaAlumnos.sacar(),tiempoAct);
                }
        }                
    }
    public void procesoNuevoAlumno(int tiempoAct,int cantUsuarios){
        Alumno alumno;
        int nroPc;
        alumno=new Alumno(cantUsuarios);
        System.out.println("Llego "+alumno);
        if(this.ColaAlumnos.estaVacia()){
            nroPc=BuscaPcLibre();
            if(nroPc!=-1){
                OcuparPc(nroPc,alumno,tiempoAct);
            }
        }    
        else{
            this.ColaAlumnos.meter(alumno);
        }
    }
}
