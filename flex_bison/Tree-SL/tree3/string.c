# include "string.h"
# include "defs.h"

/*MUDULOS PUBLICOS QUE YO AGREGUE*/

/*PUBLICOS*/

str create(){/*CREA NODO*/
	str nvo;
	nvo=NULL;
	nvo=(str)malloc(sizeof(tnodo));
	nvo->dato = '\0';
	nvo->sig = NULL;
	return nvo;
}
	
str load(){/*CARGA CADENA*/
	str cab,aux, act;
	char c;
	cab=NULL; aux=NULL; act=NULL;

	fflush(stdin);
	c=getchar();
	while(c!=EOF && c!='\n'){
		aux=create();
		aux->dato=c;
		
		if( cab==NULL) cab = aux;
		else	act->sig = aux;
		
		act=aux;
		
		c=getchar();
	}
	return cab;
}
	
str load2(const char* F){
	
	str Cab=NULL; str act=NULL;
	str nvo;
	int i=0;
	
	while(F[i]!='\0'){
		nvo=create();
		nvo->dato = F[i];
		
		if( Cab==NULL) Cab = nvo;
		else	act->sig = nvo;
		
		act=nvo;
		i++;
	}
	return Cab;
}
	
void print(str cad){
	if(cad!=NULL){
		printf("'");
		while(cad!=NULL){
			printf("%c",cad->dato);
			cad=cad->sig;
		}
		printf("'");
	}
	
}
str concat(str a,str b){
	str trav = NULL;
	
	while(a!= NULL && a->dato == '\0')
		a = a->sig;
	while(b!= NULL && b->dato == '\0')
		b = b->sig;
	
	if (a != NULL) {
	
		trav = create();  
		trav->dato = a->dato;
		trav->sig = concat(a->sig, b);
		
	} else 
		if (b != NULL) {
			trav = create();  
			trav->dato = b->dato;
			trav->sig = concat(NULL, b->sig);  
	}
	return trav;
}

str before_token(str cad1,char c){/*PARTE UNA CADENA*/
	str aux,aux2, act;
	aux=NULL;
	aux2=NULL;
	act= NULL;
	if(cad1!=NULL){
		while(cad1!=NULL && cad1->dato!=c ){
			aux = create();
			aux->dato = cad1->dato;
		
			if(aux2 == NULL) aux2 = aux;
			else	act->sig = aux;
			
			act = aux;
			cad1 = cad1->sig;
		}
	}
	return aux2;
}
str after_token (str a, char token){
	if (a == NULL) return NULL;
	
	while (a != NULL && a->dato != token)
		a = a->sig;
	
	if (a != NULL) 
		a = a->sig;
	
	return copyStr(a);
}

void freeString(str A){
	str temp=NULL;
	
	while(A!=NULL){
		temp=A;
		A=A->sig;
		free(temp);
	}
}
int compStr(str a, str b) {					/*Funcion usada en TADSet para contains*/
	if(a!=NULL && b!=NULL){
		if (a->dato != b->dato)
			return 1;
		else
			return compStr (a->sig, b->sig);
	}
	if (a!=NULL)
		return 1;
	if (b!=NULL)
		return 1;
	return 0;
}
str copyStr(str S){
	
	str Nvo = NULL;  
	if (S != NULL) {
		Nvo=create();
		Nvo->dato = S->dato; 
		Nvo->sig = copyStr(S->sig);
	}
	return Nvo;
}
int isin (str a, str b){
	if (a==NULL) return 1; 
	if (b==NULL) return 0; 
	
	if (a->dato == b->dato) {
		if (isin(a->sig, b->sig)) 
			return 1;
	}
	return isin(a, b->sig);
}
