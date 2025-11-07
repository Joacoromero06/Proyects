#include "../Assets/data.h"
tData Inicializa(int Tipo){
	
	tData nvo=(tData)malloc(sizeof(tnodo_tData));
	nvo->tipo=Tipo;
	if(Tipo==1)
		nvo->dato.cad=NULL;
	else{
		nvo->dato.data=NULL;
		nvo->dato.sig=NULL;
	}
	return nvo;
}

//Procedimientos para Entrada/Salida estructura
void AgregaCadena(tData estruct){
	if (estruct != NULL && estruct->tipo != 1) {
		tData nav = estruct->dato.data;
		if (nav != NULL) {
			if (nav->tipo != 1) {
				tData elem=Inicializa(estruct->tipo);
				tData cad=Inicializa(1);
				cad->dato.cad=load3();
				elem->dato.data=cad;
				if (estruct->tipo==2|| (estruct->tipo==3&& Pertenece(elem,estruct)==0)){
					while(nav->dato.sig!=NULL)
						nav=nav->dato.sig;
					nav->dato.sig=elem;
				}
				else
					printf("\nElemento repetido");
			} 
		}else {
			tData elem=Inicializa(estruct->tipo);
			tData cad=Inicializa(1);
			cad->dato.cad=load3();
			elem->dato.data=cad;
			estruct->dato.data = elem;
		}
	}
}
void Agrega(tData estruct,int tipo){
	tData nuevo=Inicializa(tipo);
	Carga(nuevo);
	if(estruct!=NULL&&estruct->tipo!=1){
		tData nav=estruct->dato.data;
		if(nav==NULL)
			estruct->dato.data=nuevo;
		else if(nav->tipo!=1){
			if(estruct->tipo==2||(estruct->tipo=3&&Pertenece(nuevo,estruct)==0)){
				while(nav->dato.sig!=NULL)
					nav=nav->dato.sig;
				nav->dato.sig=nuevo;
			}
			else
			  printf("\nElemento repetido");
		}
	}
}
void Carga(tData estruct){
	int opc;
	if(estruct!=NULL&&estruct->tipo!=1){
		if(estruct->dato.data==NULL){
			do{
				printf("\nIngrese 0_Salir 1_Cadena 2_Lista 3_Conjunto ----Ingreso:");
				scanf("%d",&opc);
				switch(opc){
				case 0:
					printf(" ");
				break;
				case 1:
					AgregaCadena(estruct);
				break;
				case 2:
					Agrega(estruct,2);
				break;
				case 3:
					Agrega(estruct,3);
				break;
				default:
					printf("\nIngrese un digito valido");
				break;
				}
			}while(opc!=0);
		}
	}
}
void MuestraCadena(tData cad){
	print(cad->dato.data->dato.cad);//print(cad->dato.cad) comportamiento indefinido
}
void Muestra(tData estruct){
	if(estruct!=NULL){
		tData nav=estruct->dato.data;
		if(nav!=NULL&&estruct->dato.data->tipo==1)
			MuestraCadena(estruct);
		else{
			if(estruct->tipo==2)
				printf("[");
			if(estruct->tipo==3)
				printf("{");
			if(nav!=NULL){
				while(nav->dato.sig!=NULL){
					Muestra(nav);printf(", ");
					nav=nav->dato.sig;
				}Muestra(nav);
			}
			if(estruct->tipo==2)
			   printf("]");
			if(estruct->tipo==3)
				printf("}");
		}
	}else
	   printf("\nInicialice la estructura");
}

//Funciones para modelar Conjunto
int PerteneceCadena(tData elem,tData estruct){//--
	if(elem==NULL||estruct==NULL||elem->tipo==1||estruct->tipo==1||estruct->tipo==2||elem->tipo==2)
		return 0;
	if(elem->dato.data->tipo!=1)
		return 0;
	tData nav=estruct->dato.data;
	if(nav==NULL||nav->tipo==1)
		return 0;
	int b=0;
	while(nav!=NULL&&b==0){
		b=equal( nav->dato.data->dato.cad , elem->dato.data->dato.cad );
		nav=nav->dato.sig;
	}
	return b;
}
int Pertenece(tData elem,tData estruct){
	if(elem==NULL||estruct==NULL||elem->tipo==1||estruct->tipo==1||estruct->tipo==2||elem->tipo==2)
		return 0;
	tData nav=estruct->dato.data;
	if(nav==NULL)
		return 0;
	int b=0;
	if(elem->dato.data==NULL){//elem conj vacio o lista vacia
		if(elem->tipo==3){
			while(nav!=NULL&&b==0){
				b=DobleInclucion(elem,nav);
				nav=nav->dato.sig;
			}return b;
		}
		else{
			while(nav!=NULL&&b==0){
				b=ComparaLista(elem,nav);
				nav=nav->dato.sig;
			}return b;
		}
	}
	if(elem->dato.data->tipo==1)//caso base
		return PerteneceCadena(elem,estruct);
	else{//subconj o sublista
		if(elem->tipo==3){
			while(nav!=NULL&&b==0){
				b=DobleInclucion(elem,nav);
				nav=nav->dato.sig;
			}return b;
		}
		else{
			while(nav!=NULL&&b==0){
				b=ComparaLista(elem,nav);
				nav=nav->dato.sig;
			}return b;
		}
	}
}
int Inclucion(tData conj1,tData conj2){
	if(conj1==NULL||conj2==NULL||conj1->tipo==1||conj2->tipo==1)
		return 0;
	if(conj1->dato.data==NULL)
		return 1;
	if(conj2->dato.data==NULL)
		return 0;
	if(conj1->dato.data->tipo==1||conj2->dato.data->tipo==1)
		return 0;
	tData nav=conj1->dato.data;
	int b=1;
	while(nav!=NULL&&b==1){
		b=Pertenece(nav,conj2);
		nav=nav->dato.sig;
	}printf("valor de b(Inclusion): %d\n",b);
	return b;
}
int DobleInclucion(tData conj1,tData conj2){//--
	if(Inclucion(conj1,conj2)==1&&Inclucion(conj2,conj1)==1)
		return 1;
	else
		return 0;
}	
tData Union(tData conj1,tData conj2){
	tData conj3= NULL;
	if(conj1==NULL || conj2==NULL || conj1->tipo==1||conj2->tipo==1||conj1->tipo==2||conj2->tipo==2){
		return NULL;
	}
	
	if(conj1->dato.data == NULL && conj2->dato.data == NULL ){
		conj3= Inicializa(3);
		return conj3;
	}
	
	if(conj1->dato.data == NULL){
		conj3=Copia(conj2);
		return conj3;
	}
	
	if(conj2->dato.data == NULL){
		conj3=Copia(conj1);
		return conj3;
	}
	if(	conj1->dato.data->tipo==1 || conj2->dato.data->tipo==1)//representarian cadenas
		return NULL;

	conj3=Copia(conj1);
	tData nav2=conj2->dato.data;
	tData aux;
	while(nav2!=NULL){
		aux=Copia(nav2);
		if(Pertenece(aux,conj3)==0)
			AgregaXcolaData(conj3,aux);
		nav2=nav2->dato.sig;
	}
	return conj3;
}
tData Interseccion(tData conj1,tData conj2){
	if(conj1==NULL||conj2==NULL||conj1->tipo==1||conj2->tipo==1||conj1->tipo==2||conj2->tipo==2)
		return NULL;
	if(conj1->dato.data->tipo==1||conj2->dato.data->tipo==1)
		return NULL;
	tData nuevo=Inicializa(3);
	tData nav2=conj2->dato.data;
	tData aux;
	while(nav2!=NULL){
		aux=Copia(nav2);
		if(Pertenece(aux,conj1)==1)
			AgregaXcolaData(nuevo,aux);
		nav2=nav2->dato.sig;
	}
	return nuevo;
}
tData Diferencia(tData conj1,tData conj2){
	if(conj1==NULL||conj2==NULL||conj1->tipo==1||conj2->tipo==1||conj1->tipo==2||conj2->tipo==2)
		return NULL;
	if(conj1->dato.data->tipo==1||conj2->dato.data->tipo==1)
		return NULL;
	tData nuevo=Inicializa(3);
	tData nav1=conj1->dato.data;
	tData aux;
	while(nav1!=NULL){
		if(Pertenece(nav1,conj2)==0){
			aux=Copia(nav1);
			AgregaXcolaData(nuevo,aux);
		}
		nav1=nav1->dato.sig;
	}
	return nuevo;
}
int Cardinalidad(tData conj1){//--
	if(conj1==NULL||conj1->tipo==1||conj1->tipo==2)
		return -1;
	int c=0;
	if(conj1->dato.data==NULL)
		return c;
	tData nav=conj1->dato.data;
	if(nav->tipo == 1) return -1 ;
	while(nav!=NULL){
		c++;
		nav=nav->dato.sig;
	}
	return c;
}
//  tData buscaStr(str busc,tData conj){
// 	if(conj==NULL){return NULL;}
// 	if(conj->tipo==1){return NULL;} 
// 	if(conj->dato.data==NULL){return NULL;}
// 	if(conj->dato.data->tipo==1){return NULL;}
// 	if(conj->tipo==2){return NULL;}
// 	tData nav=conj->dato.data;
// 	while (nav!=NULL){
// 		if(equal(busc,getCadena(nav))){return nav;}
// 		avanzo(&nav);
// 	}
// 	return NULL;
// }

//Ordenar
void  Ordena(tData conj){
	if(conj==NULL){return;}
	if(conj->tipo==1){return;} 
	if(conj->dato.data==NULL){return;}
	if(conj->dato.data->tipo==1){return;}
	if(conj->tipo==2){return;}
	burbuja(conj);
}
void burbuja(tData cabeza) {
    int cambiado;
    tData nav=NULL,ultimo=NULL,next=NULL,ant=NULL;
    do { 
        cambiado = 0;
		nav = cabeza->dato.data;
		next=nav->dato.sig;
        while (nav->dato.sig != ultimo) {//debug util
			/*printf("\ncmp es: %d. ",CmpData(nav,next));
			print(nav->dato.data->dato.cad);printf(" ");
			print(next->dato.data->dato.cad);*/
			if (CmpData(nav,next)==1) { 
				if(nav==cabeza->dato.data){
					cabeza->dato.data=next;
					nav->dato.sig=next->dato.sig;
					next->dato.sig=nav;
					//avance: 
					ant=next;
					next=nav->dato.sig;
				}
				else{
					ant->dato.sig=next;
					nav->dato.sig=next->dato.sig;
					next->dato.sig=nav;
					//avance
					ant=next;
					nav=next;
				}
            }
			else{
				ant=nav;
				nav=next;
				next=next->dato.sig;
			}
			cambiado = 1;

        }
        ultimo = nav;
    } while (cambiado);
}


//data->str
void data2str(str cad,tData estruct){
	int tipo=estruct->tipo;
	while (estruct!=NULL){
		if(tipo==3){
			if(estruct->dato.data==NULL){//caso: conj vacio
				AgregaXcolaPlus(&cad,"{}");
				avanzo(&estruct);
			}
			else if(estruct->dato.data->tipo==1){//caso: cadena
				//printf("Proceso agregar x cola cadenas del conjunto a *cad*: ");
				str cadena =estruct->dato.data->dato.cad;
				//print(cadena);printf("\n");
				AgregaXcolaStr(&cad,cadena);
				//printf("Cadena *cad* construyendose: ");print(cad);printf("\n");
				if(estruct->dato.sig!=NULL){AgregaXcola(&cad,',');}//caso: hay mas elementos, entonces pongo coma
				avanzo(&estruct);
			}
			else if(estruct->dato.data->tipo==3||estruct->dato.data->tipo==2){
				AgregaXcola(&cad,'{');
				//printf("*cad* antes de la recursion: ");print(cad);printf("\n");
				//printf("Recursion:\n");
				str cadEstructura=Data2Str(estruct->dato.data);
				//printf("*cadEstructura* despues de la recursion: ");print(cadEstructura);printf("\n");
				//printf("Fin de Recursion:\n");
				AgregaXcolaStr(&cad,cadEstructura);
				AgregaXcola(&cad,'}');
				//printf("*cad* despues de la recursion: ");print(cad);printf("\n");
				avanzo(&estruct);
			}
		}
		if(tipo==2){
			if(estruct->dato.data==NULL){//caso: lista vacio
				AgregaXcolaPlus(&cad,"[]");
				avanzo(&estruct);
			}
			else if(estruct->dato.data->tipo==1){//caso: cadena
				str cadena =estruct->dato.data->dato.cad;
				AgregaXcolaStr(&cad,cadena);
				if(estruct->dato.sig!=NULL){AgregaXcola(&cad,',');}//caso: hay mas elementos, entonces pongo coma
				avanzo(&estruct);
			}
			else if(estruct->dato.data->tipo==3||estruct->dato.data->tipo==2){
				AgregaXcola(&cad,'[');
				str cadEstructura=Data2Str(estruct->dato.data);
				AgregaXcolaStr(&cad,cadEstructura);
				AgregaXcola(&cad,']');
				avanzo(&estruct);
			}
		}
	}		
}
str Data2Str(tData estruct){
	if(estruct==NULL)
		return NULL;
	if(estruct->tipo==1)
		return NULL;
	str cad=create();	
	/*podemos hacer que empiece en null. Pero hay que cambiar el procedimento recursivo pasando por referencia
	en cambio con ese nodo creado empezamos a agregar las cadenas al final con ese nodo como base 
	y despues lo sacamos haciendo que cad apunte al siguiente*/
	data2str(cad,estruct);
	cad=cad->sig;
	return cad;
}


//Funciones y Procedimientos para el manejo de la Estructura
void AgregaXcolaData(tData estruct, tData nvo){
	if(estruct!=NULL&&nvo!=NULL){
		tData nav=estruct->dato.data;
		if(nav!=NULL){
			while(nav->dato.sig!=NULL)
				nav=nav->dato.sig;
			nav->dato.sig=nvo;
		}
		else
		   estruct->dato.data=nvo;
	}
}
void AgregaXcolaCad(tData estruct, str nvo){
	tData elem=NULL;
	if(estruct!=NULL&&nvo!=NULL){
		tData nav=estruct->dato.data;
		elem=Inicializa(estruct->tipo);elem->dato.data=Inicializa(1);elem->dato.data->dato.cad=cpy(nvo);
		if(nav!=NULL){
			while(nav->dato.sig!=NULL)
				nav=nav->dato.sig;
			nav->dato.sig=elem;
		}
		else
		   estruct->dato.data=elem;
	}
}
void EliminaUltimo(tData estruct){
	if(estruct!=NULL&&estruct->tipo!=1){
		tData nav=estruct->dato.data;
		tData ant=NULL;
		if(nav!=NULL){
			while(nav->dato.sig!=NULL){
				ant=nav;
				nav=nav->dato.sig;
			}
			if(ant==NULL){
				estruct->dato.data=NULL;
				free(nav);
				nav=NULL;
			}	
			else{
				free(nav);
				nav=NULL;
				ant->dato.sig=NULL;
			}	
		}
	}
}
int CmpData(tData estruct1,tData estruct2){
	
	if(estruct1==NULL&&estruct2==NULL)
		return 0;
	if(estruct1==NULL)
		return -1;
	if(estruct2==NULL)
		return 1;
	if(estruct1->tipo!=1&&estruct2->tipo!=1){
		if((estruct1->dato.data==NULL||estruct1->dato.data->tipo!=1)&&(estruct2->dato.data==NULL||estruct2->dato.data->tipo!=1))
			return 0;
		if((estruct1->dato.data==NULL||estruct1->dato.data->tipo!=1)&&estruct2->dato.data->tipo==1)
			return 1;
		if(estruct1->dato.data->tipo==1&&(estruct2->dato.data==NULL||estruct2->dato.data->tipo!=1))
			return -1;
		if(estruct1->dato.data->tipo==1&&estruct2->dato.data->tipo==1){ 
			return cmp(estruct1->dato.data->dato.cad,estruct2->dato.data->dato.cad);
		}
	}
	return 0;//no deberia pasar
}
tData Copia(tData estruct){
	if(estruct ==NULL||estruct->tipo==1)
		return NULL;
	tData copia=Inicializa(estruct->tipo);
	if(estruct->dato.data==NULL)
		return copia;
	if(estruct->dato.data->tipo==1)
		return CopiaCadena(estruct);
	tData nav=estruct->dato.data;
	tData aux;
	while(nav!=NULL){
		aux=Copia(nav);
		AgregaXcolaData(copia,aux);
		nav=nav->dato.sig;
	}
	return copia;
}
tData CopiaCadena(tData estruct){
	tData copia=Inicializa(estruct->tipo);
	tData elem=Inicializa(1);
	elem->dato.cad=cpy(estruct->dato.data->dato.cad);
	copia->dato.data=elem;
	return copia;
}
void avanzo(tData* estruct){
	if(*estruct==NULL)
		return;
	if((*estruct)->tipo==1)
		return;
	*estruct=(*estruct)->dato.sig;
}


//Funciones y Procedimientos para el manejo de las Jerarquias
str getCadena(tData elem){
	if(elem==NULL){return NULL;printf("\nError");} 
	if(elem->tipo==1){return NULL;printf("\nError2");} 
	if(elem->dato.data==NULL){return create();printf("\nError3");} 
	if(elem->dato.data->tipo==1){return cpy(elem->dato.data->dato.cad);}
	//printf("\nError5, elem es:");Muestra(elem);
	return NULL; 
}
tData getElem(str cad,int tipo){
	tData elemCad=Inicializa(1),elem=Inicializa(tipo);
	elemCad->dato.cad=cad;elem->dato.data=elemCad;
	return elem;
}
tData getEstruct_str(str cadena,int tipo){//"q0"->{"q0"}. "{q0,q1}"->{"{q0,q1}"}. "[q0,a,q1]"->{"[q0,a,q1]"}
	tData elem=getElem(cadena,tipo),estruct=Inicializa(tipo);
	estruct->dato.data=elem;
	return estruct; 
}
tData getEstruct_data(tData estructElem,int tipo){
	tData estruct=Inicializa(tipo);
	estruct->dato.data=estructElem;
	return estruct; 
}
void setLessLevel(tData* estructParseada){
	if(*estructParseada==NULL){return;printf("\nError1");}
	if((*estructParseada)->tipo==1){return;printf("\nError2");}
	if((*estructParseada)->dato.data==NULL){return;printf("\nError3");}
	if((*estructParseada)->dato.data->tipo==1){return;printf("\nError4");}
	if((*estructParseada)->dato.sig!=NULL){return;printf("\nError5");}
	tData aux=(*estructParseada);
	(*estructParseada)=(*estructParseada)->dato.data;
	aux->dato.data=NULL;free(aux);aux=NULL;
}
void setCadena(tData elem,str qi){
	if(elem==NULL){printf("\nError");return;} 
	if(elem->tipo==1){printf("\nError2");return;} 
	if(elem->dato.data==NULL){printf("\nError3");return;} 
	if(elem->dato.data->tipo==1){elem->dato.data->dato.cad=cpy(qi);return;}
}

//Funciuon para modelar Lista
int ComparaLista(tData lis1,tData lis2){
	if(lis1==NULL||lis2==NULL||lis1->tipo==1||lis2->tipo==1)
		return 0;
	if(lis1->tipo!=2||lis2->tipo!=2)
		return 0;
	if(lis1->dato.data==NULL&&lis2->dato.data==NULL)
		return 1;
	if(lis1->dato.data==NULL||lis2->dato.data==NULL)
		return 0;
	if(lis1->dato.data->tipo==1||lis2->dato.data->tipo==1)
		return 0;
	tData nav1=lis1->dato.data;
	tData nav2=lis2->dato.data;
	int b=1;
	while(nav1!=NULL&&nav2!=NULL&&b==1){
		if(nav1->dato.data->tipo==1&&nav2->dato.data->tipo==1)
			b=equal(nav1->dato.data->dato.cad,nav2->dato.data->dato.cad);
		else
			b=0;
		nav1=nav1->dato.sig;
		nav2=nav2->dato.sig;
	}
	return b;	
}
