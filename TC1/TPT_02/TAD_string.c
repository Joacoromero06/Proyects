#include "TAD_string.h"
str create(){
	str nvo=(str)malloc(sizeof(Tnodo_caracter));
	if(nvo!=NULL){
		nvo->caracter='\0';
		nvo->sig=NULL;
	}
	else
	   printf("No hay memoria para generar un nodo");
	return nvo;
}
	
str create2(char car){
	str d;
	d=create();
		d->caracter=car;
	return d;
}
	
void print(str cadena){
	if(cadena!=NULL){
		while (cadena!=NULL){
			printf("%c",cadena->caracter);
			cadena=cadena->sig;
		}
	}	
	else
	   printf("Cadena vacia");      
}
void AgregaXcola(str*cad,str nvo){
	str aux=*cad;
	if(*cad!=NULL){
		while (aux->sig!=NULL)
			aux=aux->sig;
		aux->sig=nvo;
	} 
	else
	   *cad=nvo;
}
void FreeString(str cad){	
	str aux=NULL;
	while(cad!=NULL){
		aux=cad;
		cad=cad->sig;
		aux->sig=NULL;
		free(aux);
	}
}
str load(){
	str cad=NULL;
	str aux= create();
	fflush(stdin);
	aux->caracter=getchar();
	while (aux->caracter!='\n'){
		AgregaXcola(&cad,aux);
		aux=create();
		aux->caracter=getchar();
	}
	return cad;
}
str load2(const char* cade){
	int i=0;
	str cad=NULL;
	str aux=create();
	aux->caracter=cade[0];
	while (aux->caracter!='\0'){
		AgregaXcola(&cad,aux);
		aux=create();
		i++;
		aux->caracter=cade[i];
	}
	return cad;
}
str concat(str cad1,str cad2){
	str nvo=NULL;
	str aux=NULL;
	while (cad1!=NULL){
		aux=create();
		aux->caracter=cad1->caracter;
		AgregaXcola(&nvo,aux);
		cad1=cad1->sig;
	}
	while (cad2!=NULL){
		aux=create();
		aux->caracter=cad2->caracter;
		AgregaXcola(&nvo,aux);
		cad2=cad2->sig;
	}
	return nvo;
}
str beforetoken(str cad,char tok){
	str nvo=NULL;
	str aux=NULL;
	while (cad!=NULL&&cad->caracter!=tok){
		aux = create();
		aux->caracter=cad->caracter;
		AgregaXcola(&nvo,aux);
		cad=cad->sig;
	}
	return nvo;
}
int equal(str cad1,str cad2){
	int b=0;
	if(cad1!=NULL&&cad2!=NULL){
		while(cad1!=NULL&&cad2!=NULL&&cad1->caracter==cad2->caracter){
			cad1=cad1->sig;
			cad2=cad2->sig;
		}
		if(cad1==NULL&&cad2==NULL)
			  b=1;
	}
	return b;
}
str afterToken(str cad,char tok){
	str nvo=NULL,aux=NULL;
	if(cad!=NULL){
		while(cad!=NULL&&cad->caracter!=tok)
			cad=cad->sig;
		if(cad!=NULL){
			cad=cad->sig;
			while(cad!=NULL){
				aux=create();
				aux->caracter=cad->caracter;
				AgregaXcola(&nvo,aux);
				cad=cad->sig;
			}
		}
	}
	return nvo;
}
str cpy(str cad){
	str Nvo=NULL;
	str aux=NULL;
	while(cad!=NULL){
		aux=create();
		aux->caracter=cad->caracter;
		AgregaXcola(&Nvo,aux);
		cad=cad->sig;
	}
	return Nvo;
}
int cmp(str cad1,str cad2){
	while(cad1!=NULL&&cad2!=NULL&&cad1->caracter==cad2->caracter){
		cad1=cad1->sig;
		cad2=cad2->sig;
	}
	if(cad1==NULL&&cad2==NULL)
		  return 0;
	if(cad1==NULL)
		return -1;
	if(cad2==NULL)
		return 1;
	if(cad1->caracter<cad2->caracter)
		return -1;
	else
		return 1;
}

	
int longi(str w){
	int c=0;
	while( w !=NULL){
		c++;
		w= w->sig;
	}
	return c;
}
	
str subcad(str w){
	str b=NULL;
	str aux=NULL;
	while(w->sig!=NULL){
		aux = create2(w->caracter);
		AgregaXcola(&b,aux);
		w=w->sig;
	}
	return b;
}	
	
str ultisimbolo(str w){
	while (w->sig!=NULL){
		w=w->sig;
	}
	return create2(w->caracter);
}
