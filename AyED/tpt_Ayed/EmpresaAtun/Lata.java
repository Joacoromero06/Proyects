package EmpresaAtun;

import java.util.Random;

public class Lata {
    private static int contadorLatas=1;

    private String Codigo;
    private long FechaVenc;
    private boolean esNatural;
    private int Peso;

    public Lata(String codigo, long fechaVenc, boolean esNatural, int peso) {
        this.Codigo = codigo;
        this.FechaVenc = fechaVenc;
        this.esNatural = esNatural;
        this.Peso = peso;
    }    
    public Lata(){
        Random rand = new Random();
        this.Codigo = String.format("LAT-%03d", contadorLatas++);
        this.esNatural = rand.nextBoolean();
        this.Peso = 100;
        this.FechaVenc = generarFechaVencimiento(rand);
    }
    
    public boolean iguales(Lata l2){return this.Codigo.equals(l2.Codigo);}
    public boolean mismoTipo(boolean esNatural){return this.esNatural==esNatural;}

    private long generarFechaVencimiento(Random rand) {
        int anio = rand.nextBoolean() ? 2025 : 2026;
        int mes = (anio == 2025) ? (6 + rand.nextInt(7)) : (1 + rand.nextInt(6));
        int diaMax;

        switch (mes) {
            case 4, 6, 9, 11 -> diaMax = 30;
            case 2 -> diaMax = 28;
            default -> diaMax = 31;
        }

        int dia = 1 + rand.nextInt(diaMax);

        long fecha = dia * 1000000 + mes * 10000 + anio;

        return fecha;
    }

    public String toString() {
        String str="Lata{" +"codigo='" + this.Codigo + '\'';
        if(this.esNatural)
            str+= "al Aceite";
        else
            str+= "al Natural";
        str+=", peso=" + Peso + " gramos "+", fechaVenc=" + FechaVenc +'}';
        return str;
    }

    public String getCodigo() {
        return this.Codigo;
    }
    public void setCodigo(String codigo) {
        this.Codigo = codigo;
    }
    public long getFechaVenc() {
        return this.FechaVenc;
    }
    public void setFechaVenc(long fechaVenc) {
        this.FechaVenc = fechaVenc;
    }
    public boolean getEsNatural() {
        return this.esNatural;
    }
    public void setEsNatural(boolean esNatural) {
        this.esNatural = esNatural;
    }
    public int getPeso() {
        return this.Peso;
    }
    public void setPeso(int peso) {
        this.Peso = peso;
    }

    
}
