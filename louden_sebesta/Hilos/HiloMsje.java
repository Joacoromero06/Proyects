public class HiloMsje extends Thread{
    int cant;
    String msje;

    public HiloMsje(String msje, int cant){
        this.cant = cant;
        this.msje = msje;
    }
    public void run(){
        int i;
        for(i = 0; i < this.cant; i++){
            System.out.println(this.msje + i);
        }
    }
}
