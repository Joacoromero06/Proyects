#include "pila.h"

//Funciones y Procedimientos para manejar una pila
tpila iniciar(){
	return NULL;
}
void apilar(tpila *p ,tData nvo){
	tpila nuevo = malloc(sizeof(tnodopila));
	if(nuevo!=NULL){
		nuevo->estruct = nvo;
		nuevo->sig = NULL;
		if( esVacia(*p) )
			*p = nuevo;
		else{
			nuevo->sig = *p;
			*p = nuevo;
		}
	}	
}
int esVacia(tpila p){
	if(p==NULL)
		return 1;
	else
		return 0;
}
void desapilar(tpila *p){
	if( !(esVacia(*p)) ){
		tpila aux = *p;
		*p = (*p)->sig;
		//free(aux);
	}
}
tData tope(tpila p){
	if( !(esVacia(p)) )
		return p->estruct;
	
	else 
	   return NULL;
}

//Funcion para analizar la estructura de una cadena y apartir de ella formar un tData-AF
tData parset(str cad){
	//var para manejarcontexto
	tpila pila = iniciar();
	int contexto;
	while(cad != NULL){
		if( cad->caracter == '{' || cad->caracter=='['){ //caso nuevo contexto
			switch (cad->caracter){
			case '{':
				contexto=3;
			break;
			case '[':
				contexto=2;
			break;
			default:
			break;
			}
			tData nuevo = Inicializa(contexto);
			printf("el tope antes de apilar una estructura para cargarla es: ");Muestra(tope(pila));printf("\n");
			if(!esVacia(pila))
				AgregaXcolaData(tope(pila),nuevo);
			printf("el tope despues de apilar una estructura para cargarla es: ");Muestra(tope(pila));printf("\n");
			apilar(&pila,nuevo);printf("Se apilo una estructura: %d para cargarla\n",contexto);
			printf("el nuevo tope para cargarlo es: ");Muestra(tope(pila));printf("\n");
			cad = cad->sig; 
		}
		if(cad->caracter=='}' || cad->caracter==']'){//caso cierro contexto
			if(esVacia(pila)){printf("Error1\n");return NULL;}//no deberia estar vacia
			tData top=tope(pila);
			int tipoTope=top->tipo;
			switch (cad->caracter){
			case '}':
				contexto=3;
			break;
			case ']':
				contexto=2;
			break;
			default:
			break;
			}
			if(tipoTope!=contexto){printf("Error2\n");return NULL;}//fallo sintactico {] o [}
			desapilar(&pila);printf("Se cargo una estructura tipo %d correctamente es:",contexto);Muestra(top);printf("\n");
			if(esVacia(pila)){
				if(cad->sig==NULL){return top;}//FINALIZO {..} o [...] correcto
				else{return NULL;}//dallo sintactico {..}.. o [..]..
			}
			if(tope(pila)->tipo==3){//lo que estuvimos agregando era un elemento de un conjunto, chequear unicidad
				printf("el tope antes de eliminar su ultimo elemento es: ");Muestra(tope(pila));printf("\n");
				int b=1;
				if(!esVacia(pila)){
					EliminaUltimo(tope(pila));
					tData nav=tope(pila)->dato.data;
					printf("el tope despues de eliminar su ultimo elemento es:");Muestra(tope(pila));printf("\n");
					while (nav!=NULL&&b){
						printf("Se va a comparar: ");Muestra(top);printf(" con ");Muestra(nav);printf("\n");
						//if(nav->dato.data->tipo==1){printf("nav son elementos del conjunto\n");}
						if((nav->tipo==3&&DobleInclucion(nav,top))||(nav->tipo==2&&ComparaLista(nav,top)))
							b=0;
						nav=nav->dato.sig;
					}
				}
				if(!b){printf("Error\n");}//tope era una sublista o subconjunto repetido		
				if(b){
					AgregaXcolaData(tope(pila),top);
					printf("Okey se agrego a la estructura anterior la estructura cargada: ");
					Muestra(tope(pila));printf("\n");
				}
			}
			cad = cad->sig;
		}
		if(cad!=NULL&&cad->caracter ==','){//caso avanzo o ignoro
			cad = cad->sig;
		}
		else if(cad!=NULL&&cad->caracter!='{'&&cad->caracter!='}'&& cad->caracter!='['&&cad->caracter!=']'){//caso cadena o palabra
			str palabra = NULL;
			while(cad!= NULL && cad->caracter != ',' && cad->caracter!='}'&&cad->caracter!=']'){//carga palabra
				if(cad->caracter=='['||cad->caracter=='{'){return NULL;}//error sintactico
				AgregaXcola(&palabra,cad->caracter);
				cad = cad->sig;
			}
			printf("Se cargo una palabra: ");print(palabra);printf("\n");
			contexto=tope(pila)->tipo;
			tData elem = Inicializa(contexto),elemCad=Inicializa(1);elemCad->dato.cad = palabra;;
			elem->dato.data = elemCad;
			if(contexto==2||(contexto==3&&PerteneceCadena(elem,tope(pila))!=1))
				AgregaXcolaData(tope(pila),elem);
			printf("Al tope se le agrega una palabra al final: ");Muestra(tope(pila));printf("\n");
		}
	}  
}