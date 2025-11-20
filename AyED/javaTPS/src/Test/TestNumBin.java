package Test;

import Objects2.NumBin;

public class TestNumBin {
    public static void main(String[] args) {
        NumBin bin1=new NumBin(),bin2=new NumBin(),bin3;
        bin1.CargaNumBin();bin2.CargaNumBin();
        bin3=bin1.SumaBin(bin2);
        System.out.println("La suma es: "+bin3);

    }
}
