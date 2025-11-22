#include "data.h"
#include "defs.h"

/*==========================================================================*/
/*								CONSTRUCTORES								*/
/*==========================================================================*/
tData createData(int tipo)
{
	tData nvo;
	nvo = (tData)malloc(sizeof(TNodoData));

	switch (tipo)
	{
	case STR:
		nvo->cad = create();
		break;

	case LIST:
	case SET:
		nvo->dato = NULL;
		nvo->sig = NULL;
		break;

	case INT:
		nvo->value = 0;
		break;

	case DOUBLE:
		nvo->real = 0.0;
		break;

	case BOOL:
		nvo->value = 0;
		break;

	default:
		printf("Error en createData tipo desconocido: %d\n");
		break;
	}
	nvo->tipoNodo = tipo;

	return nvo;
}

tData createStr(char *s)
{
	tData nvo = createData(STR);
	nvo->cad = load2(s);
	return nvo;
}
tData createInt(int value)
{
	tData nvo = createData(INT);
	nvo->value = value;
	return nvo;
}
tData createDouble(double real)
{
	tData nvo = createData(DOUBLE);
	nvo->real = real;
	return nvo;
}
tData createBool(char *s)
{
	tData nvo = createData(BOOL);
	for (int i = 0; s[i]; i++)
	{
		s[i] = tolower(s[i]);
	}
	if (!strcmp(s, "true"))
	{
		nvo->value = 1;
	}
	else
		nvo->value = 0;

	return nvo;
}

/*==========================================================================*/
/*								GETTER Y SETTERS							*/
/*==========================================================================*/
tData get_dato(tData a)
{
	if (get_tipo(a) == LIST || get_tipo(a) == SET)
	{
		return a->dato;
	}
	else
	{
		printf("error %d no tiene campo dato", get_tipo(a));
		return NULL;
	}
}
tData get_next(tData a)
{
	if (get_tipo(a) == LIST || get_tipo(a) == SET)
	{
		return a->sig;
	}
	else
	{
		printf("error %d no tiene campo sig", get_tipo(a));
		return NULL;
	}
}
void set_next(tData *node, tData next)
{
	(*node)->sig = next;
}
void set_dato(tData *node, tData dato)
{
	(*node)->dato = dato;
}

int get_tipo(tData a){
	return a->tipoNodo;
}

int get_value(tData a)
{
	if(!a) {
		printf("\nQuiere obtener un valor de un tdata vacio.\n");
		exit(1);
	}
	return a->value;
}
double get_real(tData a)
{
	if (get_tipo(a) != DOUBLE)
	{
		printf("error %d no tiene campo real", get_tipo(a));
		return 0;
	}
	else
	{
		return a->real;
	}
}
str get_cad(tData a)
{
	if (get_tipo(a) != STR)
	{
		printf("error %d no tiene campo cad", get_tipo(a));
		return 0;
	}
	else
	{
		return a->cad;
	}
}

/*==========================================================================*/
/*							OPERACIONES ARITMETICAS							*/
/*==========================================================================*/
tData sumaData(tData a, tData b)
{
	if(!a || !b)
	{
		printf("punteros nulos en sumaData");
		return NULL;
	}
	if (a->tipoNodo == DOUBLE && b->tipoNodo == DOUBLE)
		return createDouble(a->real + b->real);
	if (a->tipoNodo == DOUBLE && b->tipoNodo == INT)
		return createDouble(a->real + b->value);
	if (a->tipoNodo == INT && b->tipoNodo == DOUBLE)
		return createDouble(a->real + b->value);
	return createInt(a->value + b->value);
}
tData restaData(tData a, tData b)
{
	if(!a || !b)
	{
		printf("punteros nulos en restaData");
		return NULL;
	}
	if (a->tipoNodo == DOUBLE && b->tipoNodo == DOUBLE)
		return createDouble(a->real - b->real);
	if (a->tipoNodo == DOUBLE && b->tipoNodo == INT)
		return createDouble(a->real - b->value);
	if (a->tipoNodo == INT && b->tipoNodo == DOUBLE)
		return createDouble(a->real - b->value);
	return createInt(a->value - b->value);
}
tData prodData(tData a, tData b)
{
	if(!a || !b)
	{
		printf("punteros nulos en prodData");
		return NULL;
	}
	if (a->tipoNodo == DOUBLE && b->tipoNodo == DOUBLE)
		return createDouble(a->real * b->real);
	if (a->tipoNodo == DOUBLE && b->tipoNodo == INT)
		return createDouble(a->real * b->value);
	if (a->tipoNodo == INT && b->tipoNodo == DOUBLE)
		return createDouble(a->real * b->value);
	return createInt(a->value * b->value);
}
tData cocData(tData a, tData b)
{
	if(!a || !b)
	{
		printf("punteros nulos en divData");
		return NULL;
	}
	if (a->tipoNodo == DOUBLE && b->tipoNodo == DOUBLE)
		return createDouble(a->real / b->real);
	if (a->tipoNodo == DOUBLE && b->tipoNodo == INT)
		return createDouble(a->real / b->value);
	if (a->tipoNodo == INT && b->tipoNodo == DOUBLE)
		return createDouble(a->real / b->value);
	return createInt(a->value / b->value);
}

/*==========================================================================*/
/*							MODULOS BASICOS TDATA							*/
/*==========================================================================*/
void mostrarData(tData nodo)
{

	if (nodo == NULL)
		return;

	switch (nodo->tipoNodo)
	{
	case STR:
		print(nodo->cad);
		break;
	case INT:
		printf("%d", nodo->value);
		break;
	case DOUBLE:
		printf("%f", nodo->real);
		break;
	case BOOL:
		(nodo->value == 0) ? printf("false") : printf("true");
		break;
	case LIST:
		printf("[");

		tData aux;
		aux = nodo;
		while (aux != NULL)
		{
			mostrarData(aux->dato);
			if (aux->sig != NULL)
				printf(",");
			aux = aux->sig;
		}
		printf("]");

		break;
	case SET:
		printf("{");

		tData aux2;
		aux2 = nodo;
		while (aux2 != NULL)
		{
			mostrarData(aux2->dato);
			if (aux2->sig != NULL)
				printf(",");
			aux2 = aux2->sig;
		}
		printf("}");
		break;
	}
}
void agregarData(tData *cabe, tData elem)
{
	tData cab = *cabe; 
	if (!cab )
	{
		printf("No se puede agregar un elemento a un espacio de memoria no asignado");
		return;
	}
	if( get_tipo(cab)!= SET && get_tipo(cab)!= LIST )
	{
		printf("No se puede agregar un elemento a un objeto que no es list o conjunto");
		return;
	}
	if (cab->dato == NULL)
	{

		cab->dato = elem;
	}
	else
	{
		switch (cab->tipoNodo)
		{
		case STR: case INT: case DOUBLE: case BOOL:
			break;
		case SET:
		case LIST:
		{
			tData nvo;

			if (cab->tipoNodo == SET)
			{
				if (pertenece(cab, elem) == 0)
					return;
			}
			//printf("Estoy en agregarData\n");

			while (cab->sig != NULL)
				cab = cab->sig;

			nvo = createData(cab->tipoNodo);
			nvo->dato = copiarData(elem);
			cab->sig = nvo;
			break;
		}
		default: printf("caso no def en agregarData"); break;
		}
	}
	//printf("\n\ndespues de agregar elem: ");
	//mostrarData(elem);
	//printf("\nla lista cab quedo: ");
	//mostrarData(*cabe);
	//printf("\n");
}
tData copiarData(tData copiado)
{
	tData nvo = NULL;
	if (copiado == NULL)
		return NULL;
	switch (copiado->tipoNodo)
	{
	case INT:
		nvo = createInt(copiado->value);
		break;
	case DOUBLE:
		nvo = createDouble(copiado->real);
		break;
	case BOOL:
		if (get_value(copiado)){

			nvo = createBool("true");
		}
		else
			nvo = createBool("false");

			break;
	case STR:
		nvo = createData(STR);
		nvo->cad = copyStr(copiado->cad);
		break;
	case LIST:
		nvo = createData(LIST);
		nvo->dato = copiarData(copiado->dato);
		nvo->sig = copiarData(copiado->sig);
		break;
	case SET:
		nvo = createData(SET);
		nvo->dato = copiarData(copiado->dato);
		nvo->sig = copiarData(copiado->sig);
		break;
	}

	return nvo;
}
void freeData(tData descartado)
{

	if (descartado == NULL)
		return;

	switch (descartado->tipoNodo)
	{
	case STR:
		freeString(descartado->cad);
		break;
	case BOOL:
	case INT:
	case DOUBLE:
		free(descartado); break;
	case LIST:
	case SET:
		freeData(descartado->dato);
		freeData(descartado->sig);
		break;
	}

	free(descartado);
}
int Igualdad(tData A, tData B)
{

	if (A == NULL && B == NULL)
		return 0;
	if (A == NULL || B == NULL || (A->tipoNodo != B->tipoNodo))
		return 1;

	switch (A->tipoNodo)
	{
	case STR:
		return compStr(A->cad, B->cad);
		break;
	case BOOL:
	case INT:
		return !(get_value(A) == get_value(B));
	case DOUBLE:
		return !(get_real(A) == get_real(B));
	case SET:
	{
		tData auxA = A;
		while (auxA != NULL)
		{
			if (pertenece(B, auxA->dato) != 0)
				return 1;
			auxA = auxA->sig;
		}
		tData auxB = B;
		while (auxB != NULL)
		{
			if (pertenece(A, auxB->dato) != 0)
				return 1;
			auxB = auxB->sig;
		}
		return 0;
	}

	case LIST:
	{
		tData auxA = A;
		tData auxB = B;

		while (auxA != NULL && auxB != NULL)
		{
			if (Igualdad(auxA->dato, auxB->dato) != 0)
				return 1;
			auxA = auxA->sig;
			auxB = auxB->sig;
		}
		if (auxA != NULL || auxB != NULL)
			return 1;

		return 0;
	}
	default:
		return 1;
	}
}

/*==========================================================================*/
/*							OPERACIONES CONJUNTOS							*/
/*==========================================================================*/

tData Union(tData A, tData B)
{
	if (A == NULL)
		return copiarData(B);
	if (B == NULL)
		return copiarData(A);

	if (A->tipoNodo != SET || B->tipoNodo != SET)
		return NULL;

	tData C_Cab = NULL, C_act = NULL;
	tData aux;

	while (A != NULL)
	{

		aux = createData(SET);
		aux->dato = copiarData(A->dato);

		if (C_Cab == NULL)
			C_Cab = aux;
		else
			C_act->sig = aux;

		C_act = aux;

		A = A->sig;
	}
	while (B != NULL)
	{

		if (pertenece(C_Cab, B->dato))
		{

			aux = createData(SET);
			aux->dato = copiarData(B->dato);

			if (C_Cab == NULL)
				C_Cab = aux;
			else
				C_act->sig = aux;

			C_act = aux;
		}
		B = B->sig;
	}

	return C_Cab;
}
tData Interseccion(tData A, tData B)
{
	if (A == NULL || B == NULL)
		return NULL;

	if (A->tipoNodo != SET || B->tipoNodo != SET)
		return NULL;
	tData C_Cab = NULL, C_act = NULL;
	tData aux;
	while (A != NULL)
	{

		if (!pertenece(B, A->dato))
		{

			aux = createData(SET);
			aux->dato = copiarData(A->dato);

			if (C_Cab == NULL)
				C_Cab = aux;
			else
				C_act->sig = aux;

			C_act = aux;
		}
		A = A->sig;
	}
	return C_Cab;
}
tData Diferencia(tData A, tData B)
{
	if (A == NULL)
		return NULL;
	if (B == NULL)
		return copiarData(A);

	if (A->tipoNodo != SET || B->tipoNodo != SET)
		return NULL;

	tData C_Cab = NULL, C_act = NULL;
	tData aux;
	while (A != NULL)
	{

		if (pertenece(B, A->dato))
		{

			aux = createData(SET);
			aux->dato = copiarData(A->dato);

			if (C_Cab == NULL)
				C_Cab = aux;
			else
				C_act->sig = aux;

			C_act = aux;
		}
		A = A->sig;
	}
	return C_Cab;
}
tData DifSimetrica(tData A, tData B)
{
	if (A->tipoNodo != SET || B->tipoNodo != SET)
		return NULL;

	tData D = Diferencia(A, B);
	tData E = Diferencia(B, A);

	tData F = NULL;
	F = Union(D, E);

	return F;
}
int pertenece(tData A, tData elem)
{
	if (A == NULL || elem == NULL)
		return 1;
	if (A->tipoNodo != LIST && A->tipoNodo != SET)
		return 1;

	tData aux = A;

	while (aux != NULL)
	{
		if (Igualdad(aux->dato, elem) == 0)
			return 0;
		aux = aux->sig;
	}

	return 1;
}
int pertenece_completing(tData a, tData b){
	int band=1;
	while(b!=NULL && a!=NULL && band){
		if(pertenece(a,b))
			b=0;
		b=b->sig;
	}
	return band;
}
int contenido(tData A, tData B)
{

	tData auxA = A;
	while (auxA != NULL)
	{
		if (pertenece(B, auxA->dato) != 0)
			return 0;
		auxA = auxA->sig;
	}
	return 1;
}
int cardinalidad(tData A)
{

	if (A == NULL || A->tipoNodo != SET)
		return 0;

	int c = 0;

	while (A != NULL)
	{
		c++;
		A = A->sig;
	}

	return c;
}

/*==========================================================================*/
/*							OPERACIONES LISTA								*/
/*==========================================================================*/
tData elemento_pos(tData lista, int pos)
{
	if (!lista || get_tipo(lista) != LIST)
	{
		printf("Error elemento_pos es una operacion de listas");
		return NULL;
	}
	int i=1;
	while(i<pos && lista!=NULL){
		i++;
		lista=get_next(lista);
	}
	if(i=pos && lista!=NULL) return get_dato(lista);
	printf("Posicion más grande que indice.\n");
	return createStr("-1");
	

	}
tData concat_list(tData a,tData b){
	tData nuevo=createData(LIST);
	if(!a || !b){printf("error concatenar_listas  de listas nulas");return NULL;}
	printf("inicio concat list\n");
	while(a!=NULL){
		agregarData(&nuevo, get_dato(a));	
		a = get_next(a);
	}
	printf("medio concat list\n");
	while(b!=NULL)
	{
		agregarData(&nuevo, get_dato(b));
		b = get_next(b);
	}
	printf("saliendo concat list\n");
	return nuevo;	
}
void eliminar_pos(tData *l, int pos)
{
	int i;
	tData ant, nav;
	nav = *l;
	ant = NULL;

	/*if (!nav)
	{
		printf("Error eliminar_pos nav null");
		return ;
	}*/
	if (!nav || get_tipo(nav) != LIST)
	{
		printf("Error eliminar_pos es una operacion de listas");
		return ;
	}

	for (i = 1; i < pos && nav; i++)
	{
		ant = nav;
		nav = get_next(nav);
	}
	if (!nav)
		return; // posicion fuera de la lista
	if (!ant)
	{ // eliminar cabeza
		
		*l = get_next(nav);
		//freeData(get_dato(nav));
		//freeData(nav);
	}
	else
	{ // eliminacion en lista
		set_next(&ant, get_next(nav));
		freeData(nav);
	}
}

