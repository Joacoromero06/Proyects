
#include "TAD_pila.h"

typedef struct{
	
	tData Q;   //conjunto de estados
	tData Alf; //alfabeto
	tData Trans; //conjunto de transiciones
	str EstIni; //el estado inicial
	tData F;      //el conjunto de estados de aceptacion
	
} TAF ;


typedef TAF* P_TAF  ; 

void cargarAF ( TAF* A);
void mostrarAF(TAF A);
//void crearAF(A);
void CargaQDet(tData Q, int N); // carga el conj de estados del automata 
void cargaAlfabeto(tData alf, int N);
void CargaDeltaDet(tData , int N);  //carga el conjunto de funciones de transicion de un AFD
void CargaDeltaNoDet(tData delta, int N);  //carga el conjunto de funciones de transicion  de un AFND
void determinarAF(TAF );
int auxDeterminarAFD(tData delta);
void determinarCadenaAceptada(TAF A);

str deltaextendidodet(str q,str w,tData delta);

str deltadet (str q, str a,tData delta);
int compara(str q , str a , tData tup); //navega en todas las funciones de transicion hasta encontrar el caso particular
str desti(tData tup);
tData deltaNoDet (str q,str a , tData deltita);
tData destinodet(tData tup);
tData deltaExtendidoNoDet(str q,str w,tData delta);

void cargarAFConParset(TAF* A);
void cargarAFPredefinido(TAF* A);
void cargarAFNDPredefinido(TAF* A);
