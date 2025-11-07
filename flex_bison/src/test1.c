#include "af.h"
int main(){
    print(load());
    print(load());
    /*tData conj;
    conj=parset(load2("q13,q2,q1,qf,q0,q,qf"));
    printf("Conjunto: ");Muestra(conj);printf("\n");

    Ordena(conj);                   //print(conj->dato.data->dato.cad);print(conj->dato.data->dato.sig->dato.cad);print(conj->dato.data->dato.sig->dato.sig->dato.cad);    
    printf("Conjunto: ");Muestra(conj);printf("\n");

    str cad=Data2Str(conj);
    printf("Cadena correspondiente al conjunto: ");print(cad);printf("\n");*/
    //printf("*cad* 1er caracter:  ");printf("%c",cad->caracter);printf("\n");
    //printf("*cad* 2do caracter:  ");printf("%c",cad->sig->caracter);printf("\n");
    
    //prueba funciones str
    /*str cad1=NULL;
    str cad2=load2("hola");
    str cad3=NULL;

    AgregaXcolaStr(&cad1,cad2);
    printf("Cadena : ");print(cad1);printf("\n");

    AgregaXcolaPlus(&cad1,"chau");
    printf("Cadena : ");print(cad1);printf("\n");*/
    
    return 0;
}