
public class Numero{
    private int x;

    public Numero(int x){
        this.x = x;
    }

    public int getX(){
        return this.x;
    }
    public void setX(int x){
        this.x = x;
    }

    public int FactIter(){
        int fact, i;
        fact = 1;
        for (i = 1; i <= this.x; i+=1){
            fact *= i;
        }

        return fact;
    }

    public int GcdIterativo(int y){
        int x, resto;
        x = this.x;
        while ( y != 0 ){
            resto = x % y;
            x = y;
            y = resto;
        }
        return x;
    }
    public int GcdIterativo2( Numero yNumero ){
        int x, resto, y;
        x = this.x;
        y = yNumero.getX();
        while ( y != 0 ){
            resto = x % y;
            x = y;
            y = resto;
        }
        return x;
    }
    public int GcdRecursivo(int y){
        int resto;
        if (y == 0)
            return this.x;
        else{
            resto = this.x % y;
            setX(y);
            return GcdRecursivo(resto);
        }

    }
    public int GcdRecursivo2( Numero y ){
        int resto;
        if ( y.getX() == 0 ){
            return this.x;
        }
        else {
            resto = this.x % y.getX();
            setX( y.getX() );
            y.setX( resto );
            return GcdRecursivo2( y );
        }
    }


}

