#include "data.h"

typedef struct nodopila{
	tData estruct;
	struct nodopila* sig;
}tnodopila;

typedef tnodopila* tpila;

//Funciones y Procedimientos para manejar una pila
tpila iniciar(); 
void apilar(tpila*,tData);
void desapilar(tpila*);
tData tope(tpila);
int pilavacia(tpila);

//Funcion para analizar la estructura de una cadena y apartir de ella formar un tData-AF
tData parsetPlus(str);
tData parset(str); 


