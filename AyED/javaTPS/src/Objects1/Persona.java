package Objects1;

import java.util.Random;

public class Persona {
    private String Nombre, Apellido;
    private long dni;
    private char sexo;

    public Persona(String nombre, String apellido, long dni, char sexo) {
        this.Nombre = nombre;
        this.Apellido = apellido;
        this.dni = dni;
        this.sexo = sexo;
    }
    public Persona(){
        this(GenerarNombre(),GeneraApellido(),GeneraDni(),GeneraSexo());
    }
    public static String GenerarNombre(){
        String[] nombres={"Juan", "María", "Carlos", "Ana", "Luis", "Laura", "Pedro", "Sofía", "Javier", "Elena", 
"Miguel", "Isabel", "José", "Carmen", "David", "Lucía", "Daniel", "Paula", "Alejandro", "Claudia"};
        return nombres[new Random().nextInt(nombres.length)];
    }
    public static String GeneraApellido(){
        String[] apellidos={"González", "Rodríguez", "Fernández", "López", "Martínez", "Pérez", "Gómez", "Sánchez", 
"Díaz", "Alvarez", "Romero", "Suárez", "Torres", "Ruiz", "Ramírez", "Flores", "Acosta", 
"Medina", "Herrera", "Ortiz"};
        return apellidos[new Random().nextInt(apellidos.length)];
    }
    public static long GeneraDni(){
        return  10000000+new Random().nextInt(20);
    }
    public static char GeneraSexo(){
        if(new Random().nextInt(10)<5)
            return 'm';
        else 
            return 'f';
    }
    
    public boolean iguales(Persona p2){
        return this.dni==p2.dni;
    }
    public String toString(){
        return "DNI: "+this.dni+" Apellido: "+this.Apellido+" Nombre: "+this.Nombre+" sexo: "+this.sexo;
    }
}
