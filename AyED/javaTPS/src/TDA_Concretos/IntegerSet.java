package TDA_Concretos;
import CL2.AbsSet;
import TDA.Nodo;
public class IntegerSet extends AbsSet{
	
	public boolean iguales(Object objA, Object objB){
		return (int)objA==(int)objB;
	}
	public void muestra(){
		Nodo temp;	
		temp=this.conjunto;
		while (temp!=null){
			System.out.println("elemento " + ((int)temp.getNodoInfo()));
			temp=temp.getNextNodo();			
		}
	}

}


