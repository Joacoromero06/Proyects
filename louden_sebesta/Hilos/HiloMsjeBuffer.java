public class HiloMsjeBuffer extends Thread{
    String msje;
    int cant;
    WrapInt pos;
    String[] buffer;

    public HiloMsjeBuffer(String msje, int cant, String[] buffer, WrapInt pos){
        this.msje = msje;
        this.cant = cant;
        this.buffer = buffer;
        this.pos = pos;
    }
    public void run(){
        int i;
        for(i = 0; i < this.cant; i++ ){
            System.out.println(msje + ": Pos donde agrego: " + pos.value);
            this.buffer[this.pos.value] = this.msje;
            this.pos.value = this.pos.value + 1;
            System.out.println(msje + "Aumente la posicion a: " + pos.value);
        }
    }
}