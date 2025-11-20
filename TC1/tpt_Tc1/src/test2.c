#include "af.h"
int main(){
    //tData conj=parset(load2("{q0,q1,q2,q3}"));
    //printf("Conjunto: ");Muestra(conj);printf("\n");

    tData conj2=parset(load2("{q0,q1},{[q0,0,q1],[q1,1,q0]}"));
    printf("Conjunto: ");Muestra(conj2);printf("\n");
    return 0;
}