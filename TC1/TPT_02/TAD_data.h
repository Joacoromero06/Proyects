#ifndef TAD_DATA_H
#define TAD_DATA_H
#include "TAD_string.h"
typedef struct tnodo_dataType {
	int tipo;
	union {
		str cad;
		struct {
			struct tnodo_dataType* data;
			struct tnodo_dataType* sig;
		};
	} dato;
} tnodo_tData;
typedef tnodo_tData* tData;
tData Inicializa(int);
void AgregaCadena(tData);
void Agrega(tData,int);
void Carga(tData);
void Muestra(tData);
void MuestraCadena(tData); 
int PerteneceCadena(tData,tData);
int Pertenece(tData,tData);
int Inclucion(tData,tData);
int DobleInclucion(tData,tData);
int ComparaLista(tData,tData);
void AgregaXcolaData(tData,tData);
void EliminaUltimo(tData);
int Compara(tData,tData); //compara 2 conj o 2 listas
tData Copia(tData);
tData CopiaCadena(tData);
tData Union(tData,tData);
tData Interseccion(tData,tData);
tData Diferencia(tData,tData);
int Cardinalidad(tData);

tData cargaTuplaDet(); //carga una sola transicion de un AFD
tData cargaTuplaNoDet(); // carga una sola transicion de un AFND

#endif
