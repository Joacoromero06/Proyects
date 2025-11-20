#ifndef TAD_PILA_H
#define TAD_PILA_H
#include "TAD_data.h"


typedef struct nodopila{
	tData estruct;
	struct nodopila *sig;
}tnodopila;

typedef tnodopila *tpila;

tpila iniciar(); //return null
void apilar(tpila*,tData);
void desapilar(tpila*);
tData tope(tpila);
int pilavacia(tpila);
tData parset(str);



#endif
