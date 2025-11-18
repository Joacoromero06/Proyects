package Objects2;

import TDA_Concretos.ListaIntComun;
import assets.Entrada;

public class NumBin extends ListaIntComun {
    
    public void CargaNumBin(){
        System.out.println("Ingrese un numero Binario: ");
        int num=Entrada.sc.nextInt();
        while (num!=0) {
            insertar(num%10, 0);
            num/=10;
        }
    } 
    
    public int UltBit(){
        return (int)devolver(ultimo);
    }
    
    public NumBin SumaBin(NumBin B){
        
        NumBin resultado = new NumBin();
        int carry = 0,sumaParcial,suma,bitA,bitB;

        while (!this.estaVacia() || !B.estaVacia() || carry != 0) {
            bitA = 0;
            bitB = 0;
            if (!this.estaVacia()) {
                bitA = this.UltBit();
                this.eliminar(this.tamanio() - 1);  // eliminar el último
            }
            if (!B.estaVacia()) {
                bitB = B.UltBit();
                B.eliminar(B.tamanio() - 1); 
            }

            suma =bitA+bitB+carry;//a lo mucho 3
            carry=suma/2;//solo si es 3 o 2 hay acarreo de 1 "logico"
            sumaParcial=suma%2;//es el primer digito de la suma de los bits mas el acarreo 
            resultado.insertar(sumaParcial, 0); // insertamos al inicio
        }

        return resultado;
    }
    public String toString(){
        String str="Numero Binario: ";
        int bit;
        for(int i=0;i<this.tamanio();i++){
            bit=(int)this.devolver(i);
            str+=bit;
        }            
        return str;
    
    }
}
