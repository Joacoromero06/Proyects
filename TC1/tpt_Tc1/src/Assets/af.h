#include "pila.h"
typedef struct{
	tData Q;   //conjunto de estados
	tData Alf; //alfabeto
	tData Trans; //conjunto de transiciones
	str EstIni; //el estado inicial
	tData F;      //el conjunto de estados de aceptacion
} TAF ; 

//Procedimientos para Entrada/Salida AF
void CargarAF (TAF*);
void MostrarAF(TAF);
void CargaConj(tData,int); 

//Funciones y Procedimientos para AFD
void CargaDeltaDet(tData,int); 
tData CargaTuplaDet();
tData CargaTuplaDet2(str,str,str);
str DeltaExtendidoDet(str q,str w,tData delta);
str DeltaDet (str q, str a,tData delta);
str EstDest(tData tup);

// Funciones para AFND 
void CargaDeltaNoDet(tData delta, int N); 
tData CargaTuplaNoDet();
tData DeltaExtendidoNoDet(str q,str w,tData delta);
tData DeltaNoDet (str q,str a , tData delta);
tData ConjDest(tData tup);

//afnd2afd
TAF afnd2Afd (TAF);
str buscaQnoDef(tData,tData,tData);
void ordenaEstados(TAF*);
void renombrarEstados(TAF*);
void renombrarDelta(tData,str,str);
void cambiaTupla(tData,str,str);
void renombrarF(tData,str,str);
void afndToAfd(TAF);

//Funciones y Procedimientos para modelar AF
int DeterminarAFD(tData delta);
int DeterminaCadenaAceptada(TAF A,str w);
void CargarParset(TAF* A);
void CargaXcadena(TAF* A);
void CargarAFDPredefinido(TAF* A);
void CargarAFNDPredefinido(TAF* A);
void CargaXcadenaNoDet(TAF* A);
str getConjuntoTrans(str* );

int Compara(str q , str a , tData tup);
