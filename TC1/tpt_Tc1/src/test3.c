#include "af.h"
int main(){
    TAF N,D;
    CargarAFNDPredefinido(&N);
    D=Afnd2Afd(N);
    MostrarAF(D);
    renombrarEstados(&D);MostrarAF(D);
    return 0;
}