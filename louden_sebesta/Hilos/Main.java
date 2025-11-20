public class Main{
    public static void main(String args[]){
        HiloMsje hola_msje = new HiloMsje("Hola", 50);
        HiloMsje chau_msje = new HiloMsje("Chau", 50);    
        hola_msje.start();
        chau_msje.start();

    }
}