#include "TAD_data.h"
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
void AgregaCadena(tData estruct){
	if (estruct != NULL && estruct->tipo != 1) {
		tData nav = estruct->dato.data;
		if (nav != NULL) {
			if (nav->tipo != 1) {
				tData elem=Inicializa(estruct->tipo);
				tData cad=Inicializa(1);
				cad->dato.cad=load();
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
			cad->dato.cad=load();
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
	print(cad->dato.cad);
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
int PerteneceCadena(tData elem,tData estruct){
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
	}
	return b;
}
int DobleInclucion(tData conj1,tData conj2){
	if(Inclucion(conj1,conj2)==1&&Inclucion(conj2,conj1)==1)
		return 1;
	else
		return 0;
}
	
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
int Compara(tData estruct1,tData estruct2){
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
			return -1;
		if(estruct1->dato.data->tipo==1&&(estruct2->dato.data==NULL||estruct2->dato.data->tipo!=1))
			return 1;
		if(estruct1->dato.data->tipo==1&&estruct2->dato.data->tipo==1)
			return cmp(estruct1->dato.data->dato.cad,estruct2->dato.data->dato.cad); //
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
tData Union(tData conj1,tData conj2){
	tData conj3= NULL;
	if(conj1==NULL || conj2==NULL || conj1->tipo==1||conj2->tipo==1||conj1->tipo==2||conj2->tipo==2){
/*		printf("yo programo para vos mama");*/
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
	printf("yo programo para vos mama");
	if(	conj1->dato.data->tipo==1 || conj2->dato.data->tipo==1){
		printf("yo programo para vos mama");
		return NULL;
	}
/*	printf("yo programo para vos mama");*/
	tData nuevo=Copia(conj1);
	tData nav2=conj2->dato.data;
	tData aux;
	printf("");
	while(nav2!=NULL){
		aux=Copia(nav2);
		if(Pertenece(aux,nuevo)==0)
			AgregaXcolaData(nuevo,aux);
		nav2=nav2->dato.sig;
	}
	return nuevo;
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
int Cardinalidad(tData conj1){
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


