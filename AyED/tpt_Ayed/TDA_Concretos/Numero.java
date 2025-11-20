package TDA_Concretos;

//import Recursos.Entrada;
//import Recursos.Operable;

//import Recursos.ParOrdenado;

public class Numero {
    private int valor;

    ////CONSTRUCTORES SETTERS Y GETTERS


    public Numero(){
        this.valor=0;
    }
    public Numero(int valor){
        this.valor=valor;
    }
    public int GetValor(){
        return this.valor;
    }
    public void SetValor(int valor){
        this.valor=valor;
    }
    /*
    
    //METODOS DE IOPERACIONES GENERALES PARA CUALQUIER OBJETO MATEMATICO
    public void Sumar(Operable a, Operable b) {
        this.valor=((Numero)a).GetValor()+((Numero)b).GetValor();
    }
    
    
    public void Resta(Operable a, Operable b) {
        this.valor=((Numero)a).GetValor()-((Numero)b).GetValor();
    }
    
    
    public void Producto(Operable a, Operable b) {
        this.valor=((Numero)a).GetValor()*((Numero)b).GetValor();
    }
    
    
    public void Potencia(Operable a, int n) {
        this.valor=(int)Math.pow((double)((Numero)a).GetValor(),(double)n);
    }
    
    
    public boolean Iguales(Operable a, Operable b) {
        return ((Numero)a).GetValor()==((Numero)b).GetValor();
    }  
    
    
    public Operable Carga(){
        System.out.println("Carga Entero: ");

        return new Numero(Entrada.sc.nextInt());
    }
    
    
    public Operable Inicializa(){
        return new Numero();
    }
*/
    
    //METODOS RECURSIVOS DESTRUYEN LOS VALORES
    /*
    public static int MCD(Numero a, Numero b){
        if(b.GetValor()==0)
            return a.GetValor();
        else{
            a.SetValor(b.GetValor());
            b.SetValor(Resto(a, b));
            return MCD(  a  , b);
        }     
    }
    public static int RestoNaturales(Numero a, Numero b){
        if(b.GetValor()==0)
            return -1;
        if(a.GetValor()-b.GetValor()<0)
            return a.GetValor();
        else{
            a.SetValor(a.GetValor()-b.GetValor());
            return RestoNaturales(a,b);
        }
    }
    public static int CocienteNaturales(Numero a, Numero b){
        if(a.GetValor()-b.GetValor()<0||b.GetValor()==0)
            return 0;
        else{
            a.SetValor(a.GetValor()-b.GetValor());
            return CocienteNaturales(a,b)+1;
        }
    }
    public static int Resto(Numero a, Numero b){
        Numero x=new Numero(Math.abs(a.GetValor()));
        Numero y=new Numero(Math.abs(b.GetValor()));
        int r=RestoNaturales(x,y);
        if(a.GetValor()<0){
            if(b.GetValor()<0)
                r=(-1)*b.GetValor()-r;
            else
                r=b.GetValor()-r;
        }
        return r;
    }
    public static int Cociente(Numero a, Numero b){
        Numero x=new Numero(Math.abs(a.GetValor()));
        Numero y=new Numero(Math.abs(b.GetValor()));
        int c=CocienteNaturales(x,y);
        if(a.GetValor()<0){
            if(b.GetValor()<0)
                c++;
            else
                c=(-1)*c-1;
        }
        else 
            if(b.GetValor()<0)
                c*=-1;
        return c;
    } */
    //NO ES BUENA PRACTICA USAR RECURSIVIDAD CON OBJETOS
    

    //METODOS UNICOS DE LA CLASE NUMERO
    public String toString(){
        return "Numero: "+this.valor;
    }
    
    /*
    public void CocienteNaturales(Numero a, Numero b){        
        Numero cpyA=new Numero(a.GetValor());
        Numero cpyB=new Numero(b.GetValor());
        Numero coc=new Numero();
        if(b.GetValor()!=0){
            while (cpyA.GetValor()>=cpyB.GetValor()) {
                cpyA.SetValor(cpyA.GetValor()-cpyB.GetValor());
                coc.SetValor(coc.GetValor()+1);
            }
        }
        this.valor =coc.GetValor();
    }


    public void RestoNaturales(Numero a, Numero b){
        Numero cpyA=new Numero(a.GetValor());
        Numero cpyB=new Numero(b.GetValor());
        if(b.GetValor()!=0){
            while (cpyA.GetValor()>=cpyB.GetValor()) 
                cpyA.SetValor(cpyA.GetValor()-cpyB.GetValor());   
        }
        this.valor =cpyA.GetValor();  
    }


    public static ParOrdenado CRNaturales(Numero a, Numero b){
        Numero cocN=new Numero(a.GetValor());
        Numero resN=new Numero(b.GetValor());

        cocN.CocienteNaturales(a, b);
        resN.RestoNaturales(a,b);

        return new ParOrdenado(cocN, resN);
    }


    public  static ParOrdenado CyR(Numero a, Numero b){
        Numero cpyA=new Numero(Math.abs(a.GetValor()));
        Numero cpyB=new Numero(Math.abs(b.GetValor()));
        ParOrdenado crN=CRNaturales( cpyA , cpyB );
        
        Numero aux1=new Numero();
        Numero uno=new Numero(1);
        Numero muno=new Numero(-1);

        if(a.GetValor()<0){
            if(crN.getY().GetValor()!=0){

                aux1.Resta( cpyB , crN.getY() );
                crN.setY(aux1);

                if(b.GetValor()<0){
                    aux1 =new Numero();
                    aux1.Sumar( crN.getX() , uno );
                    crN.setX(aux1);
                }
                else{
                    aux1 =new Numero();
                    aux1.Resta( muno , crN.getX() );
                    crN.setX(aux1);
                }
            }
            else{
                if(b.GetValor()>0){
                    aux1.Producto(crN.getX(), muno);
                    crN.setX(aux1);
                }
            }
        }
        else{
            if(b.GetValor()<0){
                aux1.Producto( crN.getX() , muno );
                crN.setX(aux1);
            }  
        }
        return crN;
    }
    
    
    public void MCD(Numero a, Numero b){
        
        ParOrdenado cr;
        
        Numero cpyA=new Numero(Math.abs(a.GetValor()));
        Numero cpyB=new Numero(Math.abs(b.GetValor()));
        
        while (cpyB.GetValor()!=0) {
            cr=CyR( cpyA , cpyB );
            cpyA=new Numero(cpyB.GetValor());
            cpyB.SetValor(cr.getY().GetValor());
        }
        this.valor=cpyA.GetValor();
    } 


    public void MCM(Numero a, Numero b){
        
        Numero mcd=new Numero(); 
        mcd.MCD(a, b);

        Numero prod=new Numero();
        prod.Producto(a, b);
        if(prod.GetValor()<0){
            prod.SetValor(prod.GetValor()*-1);
        }

        ParOrdenado cr=CyR(prod, mcd);
        this.valor=cr.getX().GetValor();
    }*/
}
