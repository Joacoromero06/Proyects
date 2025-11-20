package assets;

public class PropNum {
    public static boolean esCapicua(int n){
        boolean b=false;
        int aux;
        int inv=0,cpy=n;
        while (cpy!=0) {
            aux=cpy%10;
            cpy/=10;
            inv=inv*10+aux;
        }
        if(inv==n)
            b=true;
        return b;
    }
}
