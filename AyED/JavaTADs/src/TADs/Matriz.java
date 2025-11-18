package TADs;

import Recursos.Entrada;

import Recursos.Operable;;

public class Matriz implements Operable{
    Operable matriz[][];
    int n,m;


    //CONSTRUCTORES SETTERS Y GETTERS
    public Matriz(){
        this.matriz=new Operable[0][0];
        this.m=0;
        this.n=0;
    }
    public Matriz(int n, int m) {
        this.matriz=new Operable[m][n];
        this.n = n;
        this.m = m;
    }
    public Operable[][] getMatriz() {
        return matriz;
    }
    public void setMatriz(Operable[][] matriz) {
        this.matriz = matriz;
    }
    public int getN() {
        return n;
    }
    public void setN(int n) {
        this.n = n;
    }
    public int getM() {
        return m;
    }
    public void setM(int m) {
        this.m = m;
    }


    //METODOS DE IOPERACIONES GENERALES PARA CUALQUIER OBJETO MATEMATICO
    
    public void Sumar(Operable a, Operable b) {//consulta
        int m=((Matriz)a).getM();
        int n=((Matriz)b).getN();
        if( IgualTam( (Matriz)a , (Matriz)b) ){
            if( IgualTam( this , (Matriz)a) ){
                for( int i=0;i<m;i++)
                    for( int j=0;j<n;j++){//como hace java para 
                        this.getMatriz()[i][j].Sumar( (((Matriz)a).getMatriz())[i][j] , (((Matriz)b).getMatriz() )[i][j]);
                        //saber que los 2 operables son de la misma clase? o no lo sabe? lo resuelve en el momento?
                        //diferencia entre resolver en tpo de ejec y la otra
                        
                        /* El operable que llama al metodo siosi es de una clase concreta xq tiene que estar inicializado 
                         * Segun la clase concreta del obj operable que llama el metodo. se ejecuta el metodo de esa clase concreta
                         * por eso si el met es abstracto todas las subclases lo deben implementar para que no de error
                         * 
                         * si los 2 operables no son de la misma clase nadie puede resolverlo al metodo
                         */
                        
                    }
            }     
            else
                System.out.println("Eror. La matriz donde Guardar la suma no es del mismo tamaño.");   
        }
        else
            System.out.println("Error. Las Matrices no se pueden operar.");
    }


    public void Resta(Operable a, Operable b) {
        int m=((Matriz)a).getM();
        int n=((Matriz)b).getN();
        if( IgualTam( (Matriz)a , (Matriz)b) ){
            if( IgualTam( this , (Matriz)a) ){
                for( int i=0;i<m;i++)
                    for( int j=0;j<n;j++){
                        this.getMatriz()[i][j].Resta( (((Matriz)a).getMatriz())[i][j] , (((Matriz)b).getMatriz() )[i][j]);
                    }
            }     
            else
                System.out.println("Eror. La matriz donde Guardar la suma no es del mismo tamaño.");   
        }
        else
            System.out.println("Error. Las Matrices no se pueden operar.");
    }


    public void Producto(Operable a, Operable b) {
        
    }

    
    public void Potencia(Operable a, int n) {
        int m=((Matriz)a).getM();
        int col=((Matriz)a).getN();
        if( IgualTam( (Matriz)a , this )){
            for( int i=0;i<m;i++)
                for( int j=0;j<col;j++){
                    this.getMatriz()[i][j].Potencia( (((Matriz)a).getMatriz())[i][j] , n);
                }
        }
        else
            System.out.println("Eror. La matriz donde Guardar la suma no es del mismo tamaño.");   
        
    }


    public boolean Iguales(Operable a, Operable b) {
        int i=0,j=0;
        boolean band=false;
        if(IgualTam( (Matriz)a , (Matriz)b )){
            while ( i<((Matriz)a).getM() && Iguales( ((Matriz)a).getMatriz()[i][j] , ((Matriz)a).getMatriz()[i][j] ) ) {
                while (j<((Matriz)a).getN() && Iguales( ((Matriz)a).getMatriz()[i][j] , ((Matriz)a).getMatriz()[i][j] ) ) {
                    j++;
                }
                i++;
            }
            if( i==((Matriz)a).getM() &&j==((Matriz)a).getN())
                band=true;
        }
        return band;
    }

    
    public Operable Carga(){//PARA APLICACION DE POLIMORFISMO EN UNA ESTRUCTURA DE OPERABLES. CARGO OPERABLES EN ELLA
        int opc;
        System.out.println("Vamos a cargar la Matriz");
        do {
            System.out.println("¿Que tipo de dato va alamacenar la matriz?");
            System.out.println("1 Entero, 2 Racional, 3 Complejo, 4 Variado");
            opc=Entrada.sc.nextInt();
        } while(opc!=0 && opc!=1 && opc!=2 && opc!=4);    

        System.out.println("Ingrese el tamaño de la Matriz");
        System.out.println("Cantidad de Filas: ");
        int m =Entrada.sc.nextInt();
        System.out.println("Cantidad de Columnas: ");
        int n =Entrada.sc.nextInt();
        
        Matriz nvo=new Matriz(m,n);

        switch (opc) {
            case 1:
                nvo.CargaMatriz(new Numero());
            break;
            case 2:
                nvo.CargaMatriz(new Racional());
            break;
            case 3:
                nvo.CargaMatriz(new Complejo());
            break;
            case 4:
                nvo.CargaMatriz();
            break;
            default:
                break;
        }

        return nvo;
    }


    public Operable Inicializa(){//POR LA PRECONDICION QUE EL OBJ QUE LLAME EL METODO EDBE ESTAR INICIALIZADO
        //Tendria que hacer un inicializa para matrices con tipos variados,
        //porque no solo guardo operables null y que el resultado de las operaciones se guarden en ella
        //porque las operaciones tienen como precondicion que sean de una clase concreta
        //si los metodos fueran static se podria? creo que no porque al ser static se llama con el nombre de la clase
        int opc;
        System.out.println("Vamos a Incializar la Matriz");
        do {
            System.out.println("¿Que tipo de dato va alamacenar la matriz?");
            System.out.println("1 Entero, 2 Racional, 3 Complejo");
            opc=Entrada.sc.nextInt();
        } while(opc!=0 && opc!=1 && opc!=2);    

        System.out.println("Ingrese el tamaño de la Matriz");
        System.out.println("Cantidad de Filas: ");
        int m =Entrada.sc.nextInt();
        System.out.println("Cantidad de Columnas: ");
        int n =Entrada.sc.nextInt();
        
        Matriz nvo=new Matriz(m,n);

        switch (opc) {
            case 1:
                nvo.IncializaMatriz(new Numero());
            break;
            case 2:
                nvo.IncializaMatriz(new Racional());
            break;
            case 3:
                nvo.IncializaMatriz(new Complejo());
            break;
            default:
                break;
        }

        return nvo;
    }
    

    //METODOS UNICOS DE LA CLASE MATRIZ
    public String toString(){
        String str="Vamos a Mostrar la Matriz: \n";
        for(int i=0;i<this.m;i++){
            for(int j=0;j<this.n;j++){
                str+= this.matriz[i][j].toString()+" ";
            } 
            str+="\n";
        }  
        str+="\n\n";
        return str;     
    }


    public static boolean IgualTam(Matriz A,Matriz B){
        return A.getM()==B.getM()&&A.getN()==B.getN();
    } 
    

    public void CargaMatriz(Operable tipo){//Carga matriz de un unico tipo
        for(int i=0;i<this.m;i++)
            for(int j=0;j<this.n;j++){
                System.out.println("Elemento [" + i + "][" + j + "]:");
                this.matriz[i][j]=tipo.Carga();
            }
    }


    public void CargaMatriz(){//Carga matriz de varios tipos
        int opc;
        Operable tipo=null;

        for(int i=0;i<this.m;i++)
            for(int j=0;j<this.n;j++){
                
                do {
                    System.out.println("¿Que tipo de dato va alamacenar el Elemento [" + i + "][" + j + "]");
                    System.out.println("1 Entero, 2 Racional, 3 Complejo");
                    opc=Entrada.sc.nextInt();
                } while (opc!=0&&opc!=1&&opc!=2);

                switch (opc) {
                case 1:
                    tipo=new Numero();
                break;
                case 2:
                    tipo=new Racional();
                break;
                case 3:
                    tipo=new Complejo();
                break;
                default:
                    break;
                }

                this.matriz[i][j]= tipo.Carga();
            }
    }

    
    public void IncializaMatriz(Operable tipo){
        for(int i=0;i<this.m;i++)
            for(int j=0;j<this.n;j++){
                this.matriz[i][j]=tipo.Inicializa();
            }
    }
    
     
}