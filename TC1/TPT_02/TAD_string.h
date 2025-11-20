#ifndef TAD_STRING_H
#define TAD_STRING_H
#include <stdio.h>
#include<stdlib.h>
typedef struct nodo_caracter{
	char caracter;
	struct nodo_caracter* sig;
}Tnodo_caracter;

typedef Tnodo_caracter* str;

void FreeString(str);
str create();
void print(str);
str load();
str load2(const char*);
str concat(str,str);
str beforetoken(str,char);
void AgregaXcola(str*,str);
int equal(str,str);
str afterToken(str,char);
str cpy(str);
int cmp(str,str);
/*int cmp(str cad1,str cad2)*/
str create2(char car);

int longi(str w);
str subcad(str w);
str ultisimbolo(str w);
#endif
