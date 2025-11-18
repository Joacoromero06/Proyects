package Test;

import Objects1.Libro;
import Objects1.wrapperMochila;
import Objects2.ListaLibros;
import Objects2.MochilaLibros;


public class TestProbMochila {
    public static void main(String[] args) {
        MochilaLibros mochila=new MochilaLibros();
        wrapperMochila mejor=new wrapperMochila(new MochilaLibros(mochila.GetCapacidad()));
        ListaLibros biblioteca=new ListaLibros();biblioteca.GeneraLista();
        
        System.out.println(biblioteca);System.out.println("La capacidad de mi mochila es: "+mochila.GetCapacidad());
        BPproblem(mochila,biblioteca,0,mejor);
        System.out.println(mejor.m);
        /*System.out.println(biblioteca);System.out.println("La capacidad de mi mochila es: "+mochila.GetCapacidad());
        System.out.println(BPproblem(mochila,biblioteca,0));*/

    }

    public static void BPproblem(MochilaLibros actual,ListaLibros lista,int nro,wrapperMochila mejor){
        if(nro==lista.tamanio()){
            if(actual.esMejor(mejor.m)){mejor.m=actual.Clonar();}
        }
        else{
            Libro libro=(Libro)(lista.devolver(nro));
            if(actual.GetCapacidad()>=libro.GetPeso()){
                actual.meter(libro);
                BPproblem(actual, lista, nro+1, mejor);
                actual.sacar(libro);
            }
            BPproblem(actual, lista, nro+1, mejor);
        }
    }
    public static MochilaLibros BPproblem(MochilaLibros actual, ListaLibros lista, int nro){
        if(nro==lista.tamanio()){
            return actual;
        }
        else{
            MochilaLibros Incluido,Excluido;
            Libro libro=(Libro)(lista.devolver(nro));
            Excluido=BPproblem(actual, lista, nro+1);
            if(actual.GetCapacidad()>=libro.GetPeso()){
                MochilaLibros Nueva=actual.Clonar();Nueva.meter(libro);
                Incluido=BPproblem(Nueva, lista, nro+1);
            }
            else{
                Incluido = actual;
            }
            if(Incluido.esMejor(Excluido)){
                return Incluido;
            }
            else{
                return Excluido;
            }

        }
    }
}
