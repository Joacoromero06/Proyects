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


tData createData(int);
tData createList();
tData createSet();
tData createStr();
tData createInt(int);
tData createDouble(double);
tData createBool();


//funciones nucleo
int Igualdad(tData, tData);
tData copiarData(tData);
void mostrarData(tData);
void agregarData(tData*, tData);
void freeData(tData);
tData Cargar();
tData crear_arbol(int, int);
tData toString(tData);

//operaciones de conjunto
tData Union (tData, tData);
tData Interseccion (tData, tData);
tData Diferencia (tData, tData);
tData DifSimetrica (tData, tData);
int pertenece(tData,tData);
int cardinalidad (tData);
int contenido (tData, tData);
tData toSetWToken(tData, char);

//funcion de cargar por cadena
tData crearDesdeCadena(const char*);

//funciones auxiliares de AF
char dataAcadena(tData);
tData obtener_elemento(tData, int);



tData get_dato (tData);
tData get_next (tData);

#endif
