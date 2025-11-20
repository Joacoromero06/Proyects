public class Main1{
    public static void main(String args[]){
        WrapInt pos_ini = new WrapInt(0);
        String[] buffer = new String[100];
                                        
        HiloMsjeBuffer hola_msje_buff = new HiloMsjeBuffer("Hola", 50, buffer, pos_ini );
        HiloMsjeBuffer chau_msje_buff = new HiloMsjeBuffer("Chau", 50, buffer, pos_ini);    
        hola_msje_buff.start();
        chau_msje_buff.start();
        
        for(int i = 0; i < 100; i++){
            System.out.println(buffer[i] + " " + i);
        }
    }
}