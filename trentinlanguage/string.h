#ifndef STRING_H
#define STRING_H


typedef struct nodo_str{
	char dato;
	struct nodo_str *sig;
}tnodo;
typedef tnodo *str;

/*MUDULOS PUBLICOS QUE YO AGREGUE*/

/*PUBLICOS*/
str create();
str load();
str load2(const char*);
void print(str);
str concat(str,str);
str before_token(str,char);
str after_token (str, char);

void freeString(str);
int compStr(str, str);
str copyStr(str);
int isin (str, str);

#endif
