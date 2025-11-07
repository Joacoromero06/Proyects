#include <stdio.h>
#include <stdlib.h>
typedef struct nodo_caracter{
	char caracter;
	struct nodo_caracter* sig;
}Tnodo_caracter;

typedef Tnodo_caracter* str;

//Funciones y Procedimientos para Modelar SLL
str create();
str create2(char car);
void print(str);
str load();
str load3();
str load2(const char*);
void AgregaXcola(str*,char);
void AgregaXcolaPlus(str*,const char*);
void AgregaXcolaStr(str*,str);

//Procedimiento para controlar el manejo de memoria
void FreeString(str);

//Funciones para modelar Cadena
str concat(str,str);
str beforetoken(str,char);
int equal(str,str);
str afterToken(str,char);
str cpy(str);
int cmp(str,str);

//Funciones de Cadena para el AF
int longi(str w);
str subcad(str w);
str ultisimbolo(str w);
str armaCad(const char*,int);