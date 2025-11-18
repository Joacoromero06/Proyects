#include "../Assets/str.h"

//Funciones y Procedimientos para Modelar SLL
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
	str nvo;
	nvo=create();
		nvo->caracter=car;
	return nvo;
}	
void print(str cadena){
	if(cadena!=NULL){
		printf("%c",'"');
		while (cadena!=NULL){
			printf("%c",cadena->caracter);
			cadena=cadena->sig;
		}
		printf("%c",'"');
	}	
	else
	   printf("Cadena vacia");      
}
str load3(){
	str cad=NULL; 
	int ca;
	while ((ca = getchar()) != '\n' && ca != EOF);
	char c=getchar();
	while (c!='\n'){
		AgregaXcola(&cad,c);
		c=getchar();
	}
	return cad;
}
str load(){
    str str1= NULL;
    char letras;
    //printf(" ingrese: ");
    letras= getchar(); //si tu buffer esta limpio, pide que ingreses una cadena de 0 o mas caracteres,
    // toma el primero y lo almacena en la variable letras 

    while (letras != '\n' ){
        if( str1 == NULL){
            str1= create();
            str1->caracter= letras;
        }
        else{
            AgregaXcola(&str1, letras);
        }
        letras= getchar();
    }
    return str1;
}
str load2(const char* cade){
	int i=0;
	str cad=NULL;
	char c= cade[0];
	while (c!='\0'){
		AgregaXcola(&cad,c);
		i++;
		c=cade[i];
	}
	return cad;
}
void AgregaXcab(str*cad,char c){
	if(*cad==NULL)
		*cad=create2(c);
	else{
		str aux=*cad;
		*cad=create2(c);(*cad)->sig=aux;
	}	
}
void AgregaXcola(str*cad,char c){
	str nvo=create2(c);
	str aux=*cad;
	if(*cad!=NULL){
		while (aux->sig!=NULL)
			aux=aux->sig;
		aux->sig=nvo;
	} 
	else
	   *cad=nvo;
}
void AgregaXcolaStr(str*cad,str final){
	str aux=*cad;
	if(*cad!=NULL){
		while (aux->sig!=NULL)
			aux=aux->sig;
		aux->sig=cpy(final);
	} 
	else
	   *cad=cpy(final);
}
void AgregaXcolaPlus(str*cad,const char* cade){
	str cadFinal=load2(cade);
	AgregaXcolaStr(cad,cadFinal);
}

//Procedimiento para controlar el manejo de memoria
void FreeString(str cad){	
	str aux=NULL;
	while(cad!=NULL){
		aux=cad;
		cad=cad->sig;
		aux->sig=NULL;
		free(aux);
	}
}

//Funciones para modelar Cadena
str concat(str cad1,str cad2){
	str nvo=NULL;
	while (cad1!=NULL){
		AgregaXcola(&nvo,cad1->caracter);
		cad1=cad1->sig;
	}
	while (cad2!=NULL){
		AgregaXcola(&nvo,cad2->caracter);
		cad2=cad2->sig;
	}
	return nvo;
}
str beforetoken(str cad,char tok){
	str nvo=NULL;
	while (cad!=NULL&&cad->caracter!=tok){
		AgregaXcola(&nvo,cad->caracter);
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
	str nvo=NULL;
	if(cad!=NULL){
		while(cad!=NULL&&cad->caracter!=tok)
			cad=cad->sig;
		if(cad!=NULL){
			cad=cad->sig;
			while(cad!=NULL){
				AgregaXcola(&nvo,cad->caracter);
				cad=cad->sig;
			}
		}
	}
	return nvo;
}
str cpy(str cad){
	str Nvo=NULL;
	while(cad!=NULL){
		AgregaXcola(&Nvo,cad->caracter);
		cad=cad->sig;
	}
	return Nvo;
}
int cmp(str cad1,str cad2){//-1 cad1<cad2
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

//Funciones de Cadena para el AF
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
	while(w->sig!=NULL){
		AgregaXcola(&b,w->caracter);
		w=w->sig;
	}
	FreeString(w);
	return b;
}		
str ultisimbolo(str w){
	while (w->sig!=NULL){
		w=w->sig;
	}
	return create2(w->caracter);
}
str armaCad(const char* palabra,int num){
	char c;int dig;
	str cadPalabra=load2(palabra),cadNum=NULL;
	while(num>0){
		dig=num%10;num=num/10;
		c = dig + '0';  // Resultado: 'dig'
		AgregaXcab(&cadNum,c);
	}
	return concat(cadPalabra,cadNum);
}
