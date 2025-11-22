#ifndef DATA_H
#define DATA_H

# include "string.h"

typedef struct nodo_data{
	int tipoNodo;
	union{
		str cad;
		int value;
		double real;
		struct{
			struct nodo_data * dato;
			struct nodo_data * sig;
		};
	};
}TNodoData;

typedef TNodoData * tData;

// Constructores
tData createData(int);

tData createStr(char*);
tData createInt(int);
tData createDouble(double);
tData createBool(char*);

// getters setters
tData get_dato (tData);
tData get_next (tData);
void set_dato(tData*, tData);
void set_next(tData*, tData);

int get_tipo(tData);

int get_value(tData);
double get_real(tData);
str get_cad(tData);

// Operaciones Artimeticas de tData
tData sumaData(tData, tData);
tData restaData(tData, tData);
tData prodData(tData, tData);
tData cocData(tData, tData);

// Procedimientos basicos de tData
int Igualdad(tData, tData);
void mostrarData(tData);
void agregarData(tData*, tData);
void freeData(tData);
tData copiarData(tData );

// Operaciones de conjunto
tData Union (tData, tData);
tData Interseccion (tData, tData);
tData Diferencia (tData, tData);
tData DifSimetrica (tData, tData);
int pertenece(tData,tData);
int cardinalidad (tData);
int contenido (tData, tData);

// Operaciones de lista
tData elemento_pos(tData, int pos);
void eliminar_pos(tData*, int pos);
tData concat_list(tData,tData);
int pertenece_completing(tData, tData);

#endif
