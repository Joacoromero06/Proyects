#include "str.h"
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

//Procedimientos para Entrada/Salida la estructura
void AgregaCadena(tData);
void Agrega(tData,int);
void Carga(tData);
void Muestra(tData);
void MuestraCadena(tData); 

//Funciones para modelar Conjunto
int PerteneceCadena(tData,tData);
int Pertenece(tData,tData);
int Inclucion(tData,tData);
int DobleInclucion(tData,tData);
tData Union(tData,tData);
tData Interseccion(tData,tData);
tData Diferencia(tData,tData);
int Cardinalidad(tData);
void Ordena(tData);
void burbuja(tData);
//data->str
str Data2Str(tData);

//Funciones y Procedimientos para el manejo de la Estructura
void AgregaXcolaData(tData,tData);
void AgregaXcolaCad(tData,str);
void EliminaUltimo(tData);
int CmpData(tData,tData); 
tData Copia(tData);
tData CopiaCadena(tData);
void avanzo(tData*); 

////Funciones y Procedimientos para el manejo de las Jerarquias
str getCadena(tData);
tData getElem(str,int);
tData getEstruct_str(str,int);
tData getEstruct_data(tData,int);
void setLessLevel(tData*);
void setCadena(tData,str);


//Funciuon para modelar Lista
int ComparaLista(tData,tData);
