package Test;

import Objects2.Personas;

public class TestSubsidios {
    public static void main(String[] args) {
        Personas S1=new Personas(),S2=new Personas(),S3=new Personas();
        S1.cargaPersonas();S2.cargaPersonas();S3.cargaPersonas();
        Personas OrgS1=S1.Clonar(),OrgS2=S2.Clonar(),OrgS3=S3.Clonar();
        System.out.println("Subsidio 1: "+S1);System.out.println("Subsidio 2: "+S2);System.out.println("Subsidio 3: "+S3);
        S1=(Personas) S1.Diferencia(S2.Union(S3));
        S2=(Personas)S2.Diferencia(S1.Union(S3));
        S3=(Personas)S3.Diferencia(S1.Union(S2));
        if(!OrgS1.Inclusion(S1)){System.out.println("Se eliminaron del subsidio 1: "+OrgS1.Diferencia(S1));}
        if(!OrgS2.Inclusion(S2)){System.out.println("Se eliminaron del subsidio 2: "+OrgS2.Diferencia(S2));}
        if(!OrgS3.Inclusion(S3)){System.out.println("Se eliminaron del subsidio 3: "+OrgS3.Diferencia(S3));}
        System.out.println("Subsidio 1: "+S1);System.out.println("Subsidio 2: "+S2);System.out.println("Subsidio 3: "+S3);
        
    }
}