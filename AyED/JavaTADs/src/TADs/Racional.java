package TADs;

import Recursos.Operable;

import Recursos.ParOrdenado;

public class Racional implements Operable {
    private Numero num,den;

    //CONSTRUCTORES SETTERS Y GETTERS
    public Racional(){
        this(new Numero(), new Numero(1));
    }
    public Racional(Numero num, Numero den) {
        this.num = num;
        this.den = den;
    }
    public Numero getNum() {
        return num;
    }
    public void setNum(Numero num) {
        this.num = num;
    }
    public Numero getDen() {
        return den;
    }
    public void setDen(Numero den) {
        this.den = den;
    }


    //METODOS DE IOPERACIONES GENERALES PARA CUALQUIER OBJETO MATEMATICO
    public void Sumar(Operable a, Operable b) {//consulta
        
        Numero aux1=new Numero();
        aux1.Producto( ((Racional)a).getNum()  ,  ((Racional)b).getDen() );

        Numero aux2=new Numero();
        aux2.Producto( ((Racional)b).getNum()  ,  ((Racional)a).getDen() );
        
        Numero num=new Numero();
        num.Sumar(aux1,aux2);

        Numero den=new Numero();
        den.Producto( ((Racional)a).getDen()  ,  ((Racional)b).getDen() );

        Racional aux=new Racional(num, den);
        aux.Simplificacion();
        
        this.num=aux.getNum();
        this.den=aux.getDen();
    }


    public void Resta(Operable a, Operable b) {

        Numero aux1=new Numero();
        aux1.Producto( ((Racional)a).getNum()  ,  ((Racional)b).getDen() );

        Numero aux2=new Numero();
        aux2.Producto( ((Racional)b).getNum()  ,  ((Racional)a).getDen() );
        
        Numero num=new Numero();
        num.Resta(aux1,aux2);

        Numero den=new Numero();
        den.Producto( ((Racional)a).getDen()  ,  ((Racional)b).getDen() );

        Racional aux=new Racional(num, den);
        aux.Simplificacion();
        
        this.num=aux.getNum();
        this.den=aux.getDen();
    }

  
    public void Producto(Operable a, Operable b) {
        
        Numero num=new Numero();
        num.Producto( ((Racional)a).getNum()  ,  ((Racional)b).getNum() );

        Numero den=new Numero();
        den.Producto( ((Racional)a).getDen()  ,  ((Racional)b).getDen() );

        Racional aux=new Racional(num, den);
        aux.Simplificacion();
        
        this.num=aux.getNum();
        this.den=aux.getDen();
    }


    public void Potencia(Operable a, int n) {

        Numero num=new Numero();
        num.Potencia( ((Racional)a).getNum()  ,  n );

        Numero den=new Numero();
        den.Potencia( ((Racional)a).getDen()  ,  n );

        Racional aux=new Racional(num, den);
        aux.Simplificacion();
        
        this.num=aux.getNum();
        this.den=aux.getDen();
    }

    
    public boolean Iguales(Operable a, Operable b) {//verificar
        
        ((Racional)a).Simplificacion();
        ((Racional)b).Simplificacion();

        return Iguales(((Racional)a).getNum(),((Racional)b).getNum()) && Iguales(((Racional)a).getDen(),((Racional)b).getDen());
    }


    public Operable Carga(){//consulta
        System.out.println("Carga Racional: ");

        Numero aux=new Numero();//usado solo apra llamar a Carga ¿ineficiente?¿mal diseño?

        Numero num=(Numero)aux.Carga();//aux tipo Numero llama a su carga
        Numero den=(Numero)aux.Carga();

        Racional nvo=new Racional(num,den);
        nvo.Simplificacion();

        return nvo;
    }
    
    
    public Operable Inicializa(){//¿necesario por el diseño?
        return new Racional();
    }
   
   
    //METODOS UNICOS DE LA CLASE RACIONAL
    public String toString(){
        return this.num +" / "+this.den;
    }


    public void Simplificacion(){
        
        Numero mcd=new Numero();
        mcd.MCD(this.num , this.den);

        ParOrdenado cr=Numero.CyR(this.num, mcd);
        this.num=cr.getX();

        cr=Numero.CyR(this.den, mcd);
        this.den=cr.getX();
    }

    

}

