package Test;

import EmpresaAtun.Empresa; 

public class testLogisticaEmpresa {
    public static void main(String[] args) {
        Empresa empresaAtun=new Empresa(); 
        empresaAtun.gestionTrabaja_LyM();
        empresaAtun.logisticaTrabaja_LyM();
        System.out.println(empresaAtun.getRepartidor());
        
        /*empresaAtun.mapa.CalculaRutaH_Ida(9);
        System.out.println("\n\nLA RUTA DE IDA DESDE LA FABRICA A 9 ES:");
        System.out.println(empresaAtun.mapa.getRutaHamiltoniana());
        empresaAtun.mapa.CalculaRutaH_Vuelta(9);
        System.out.println("\n\nLA RUTA DE VUELTA DESDE 9 A LA FABRICA ES:");
        System.out.println(empresaAtun.mapa.getRutaHamiltoniana());*/
        
        //empresaAtun.logisticaTrabaja();
        //empresaAtun.muestraSituacion_Logistica();
    }
}
