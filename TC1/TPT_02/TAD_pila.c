#include "TAD_pila.h"

tpila iniciar(){
	return NULL;
}
void apilar(tpila *p ,tData nvo){
	tpila nuevo = malloc(sizeof(tnodopila));
	if(nuevo!=NULL){
		nuevo->estruct = nvo;
		nuevo->sig = NULL;
		if( pilavacia(*p) )
			*p = nuevo;
		else{
			nuevo->sig = *p;
			*p = nuevo;
		}
	}
	
}
int pilavacia(tpila p){
	if(p==NULL)
		return 1;
	else
		return 0;
}
void desapilar(tpila *p){
	if( !(pilavacia(*p)) ){
		tpila aux = *p;
		*p = (*p)->sig;
		free(aux);
	}
}
tData tope(tpila p){
	
	if( !(pilavacia(p)) ){
		
		return p->estruct;
	}
	else 
	   return NULL;
}
	
tData parset(str cad){
	//var para manejarcontexto
	tpila pila = iniciar();
	tData raiz = Inicializa(3);
	apilar(&pila,raiz);
	int tipo;
	//verificar si perte
	tData top,nav;
	int b;
	//var para cargar letraxletra
	str aux,palabra;
	int contexto;
	tData nuevo;
	while(cad != NULL){
		//caso nuevo contexto
		if( cad->caracter == '{' || cad->caracter=='['){ 
			if(cad->caracter == '{')
				contexto=3;
			else
				contexto=2;
			nuevo = Inicializa(contexto);
			AgregaXcolaData(tope(pila),nuevo);
			apilar(&pila,nuevo);
			cad = cad->sig;//
		}
		
		//caso cierro contexto
		if(cad->caracter=='}' || cad->caracter==']'){
			if(pilavacia(pila))
				return NULL;
			if(cad->caracter=='}')
				tipo=3;
			else
				tipo=2;
			contexto=tope(pila)->tipo;
			if(contexto==tipo){
				top=tope(pila);
				desapilar(&pila);//printf("\nSe desapilo un nuevo contexto");
			}
			else
			   return NULL;
			if(pilavacia(pila))//faltaria cerrar raiz
				return NULL;
			if(tope(pila)->tipo==3){//lo tenemos que eliminar si ya estaba
				nav=tope(pila)->dato.data;
				b=0;
				//Muestra(nav);
				while(nav->dato.sig!=NULL&&b==0){
					if((nav->tipo==3&&DobleInclucion(nav,top)==1)||(nav->tipo==2&&ComparaLista(nav,top)==1))
						b=1;
					nav=nav->dato.sig;
				}
				if(b==1)
					  EliminaUltimo(tope(pila));
				
			}
			cad = cad->sig;
		}
		
		if(cad!=NULL&&cad->caracter ==','){
			cad = cad->sig;
		}
		else if(cad!=NULL&&cad->caracter!='{'&&cad->caracter!='}'&& cad->caracter!='['&&cad->caracter!=']'){
			palabra = NULL;
			aux=NULL;
			contexto= tope(pila)->tipo;//printf("\ntipo %d",contexto);
			
			while(cad!= NULL && cad->caracter != ',' && cad->caracter!='}'&&cad->caracter!=']'){
				aux = create();
				aux->caracter = cad->caracter;
				AgregaXcola(&palabra,aux);
				cad = cad->sig;
			}
			tData elemcad = Inicializa(1);
			elemcad->dato.cad = palabra;
			tData elem = Inicializa(contexto);
			elem->dato.data = elemcad;
			if(contexto==2||(contexto==3&&PerteneceCadena(elem,tope(pila))!=1))
				AgregaXcolaData(tope(pila),elem);
		}
		
	}
	if(!pilavacia(pila))
		  desapilar(&pila);
	else 
		return NULL;
	if(pilavacia(pila))
		return raiz;
	else {
		return NULL;
	}
}
