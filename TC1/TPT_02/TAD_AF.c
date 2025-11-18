#include <stdio.h>
#include "TAD_AF.h"


void cargarAF( TAF* A){
	int N, opcion;
	printf("\n ingrese el conjunto de estados del automata Q: ");
	A->Q=  Inicializa(3);
	printf("\n cuantos estados desea cargar: ");
	scanf("%d", &N);
	CargaQDet(A->Q, N);   
	
	printf("\n ingrese el conjunto de simbolos, (alfabeto): ");
	A->Alf= Inicializa(3);
	printf("\n cuantos simbolos desea cargar: ");
	scanf("%d", &N);
	CargaQDet(A->Alf, N); 
	
	printf("\n ingrese las funciones de transicion : ");
	A->Trans= Inicializa(3);
	printf("\n cuantas transiciones desea cargar: ");
	scanf("%d", &N);
	
	printf("\n ingrese 1 si desea cargar un AFD y otro numero distinto si desea cargar un AFND : ");
	scanf("%d", &opcion);
	if(opcion == 1){
		CargaDeltaDet(A->Trans, N);
	}
	else{
		CargaDeltaNoDet(A->Trans, N); 
	}
	
	printf("\n ingrese el estado inicial : ");
	A->EstIni= load();  
	
	printf("\n ingrese el conjunto de estados de Aceptacion: ");
	A->F= Inicializa(3);
	printf("\n cuantos estados de aceptacion desea cargar: ");
	scanf("%d", &N);
	CargaQDet(A->F, N);    
	
}

void mostrarAF (TAF A){
	
	printf("\n el conjunto de estados del automata Q: ");
	Muestra(A.Q);   
	
	printf("\n conjunto de simbolos, (alfabeto): ");
	Muestra(A.Alf);  
	
	printf("\n funciones de transicion : ");
	Muestra(A.Trans);  
	
	printf("\n estado inicial : ");
	print(A.EstIni);  
	
	printf("\n conjunto de estados de Aceptacion: ");
	Muestra(A.F);  
	
}

void CargaQDet(tData Q, int N){
	int i;
	for(i=1; i<= N; i++ ){
		printf(" ");
		AgregaCadena(Q);
	}
}	

void cargaAlfabeto( tData alf, int N){
	int i;
	for(i=1; i<= N; i++ ){
		printf(" ");
		AgregaCadena(alf);
	}
}

void CargaDeltaDet(tData delta, int N){
	int i;
	tData tupla= NULL;
	
	for(i=1; i<= N; i++ ){
		tupla= cargaTuplaDet();
		AgregaXcolaData(delta, tupla);
	}	
}

tData cargaTuplaDet(){
	
	tData tupla= Inicializa(2);
	printf("\n ingrese estado actual: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese simbolo: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese estado destino: ");
	AgregaCadena(tupla);
	
	return tupla;
}

	
			// Funciones para AFND //
	
void CargaDeltaNoDet(tData delta, int N){
	int i;
	tData tupla= NULL;
	
	for(i=1; i<= N; i++ ){
		tupla= cargaTuplaNoDet();
		AgregaXcolaData(delta, tupla);
	}	
}
	
tData cargaTuplaNoDet(){
	
	tData tupla= Inicializa(2) , conjDest= Inicializa(3) ;
	int N;
	
	printf("\n ingrese estado actual: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese simbolo: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese la cantidad de estados destino: ");
	scanf("%d", &N);
	CargaQDet(conjDest, N);
	AgregaXcolaData(tupla , conjDest);
	
	return tupla;
}
	
	
void determinarAF(TAF A){
	int rta;
	
	rta= auxDeterminarAFD(A.Trans);
	
	if(rta== -1){
		printf("\n es determinista ");
	}
	else{
		printf("\n no es determinista ");
	}
}
	
int auxDeterminarAFD(tData delta){
	tData nav= NULL, nav2= NULL; 
	if(delta != NULL){
		nav= delta->dato.data;
		if(nav != NULL){
			nav2= nav->dato.data;
			while( nav2->dato.sig != NULL ){
				nav2= nav2->dato.sig ;
			}
			return Cardinalidad(nav2);
		}
		return -1; //no pasa
	}
	return -1; //no pasa
}	

void determinarCadenaAceptada(TAF A){
	str w= NULL, q=NULL;
	tData P= NULL;
	tData estFinal= Inicializa(3) , interseccionDeEstados= NULL ;
	
	printf("\n ingresa la cadena: ");
	w= load();
	
	if(auxDeterminarAFD( A.Trans ) == -1 ){ //es determinista
		q= deltaextendidodet(A.EstIni, w, A.Trans);
		estFinal->dato.data= Inicializa(1);
		estFinal->dato.data->dato.cad= q ;
		
		if( Pertenece(estFinal ,A.F) == 1 ) {
			printf("\n la cadena es aceptada");
		}
		else{
			printf("\n la cadena no es aceptada");
		}
	}
	else{
		P = deltaExtendidoNoDet(A.EstIni, w, A.Trans);
		
		interseccionDeEstados= Interseccion( P , A.F );
		if( Cardinalidad(interseccionDeEstados) != 0   ){
			printf("\n la cadena es aceptada");
		}
		else{
			printf("\n la cadena no es aceptada");
		}
		
	}
	
	
}
	
str deltaextendidodet(str q,str w,tData delta){
	if(longi(w)==0){
		//bien
		return q;
	}
	else{
		str v,a;
		v=subcad(w);
/*		print(v);*/
		a=ultisimbolo(w);
/*		print(a);*/
		return deltadet(deltaextendidodet(q,v,delta),a,delta);
	}
}
	
	
str deltadet (str q, str a,tData delta){
	tData nav=delta->dato.data;
	while (nav!=NULL && compara(q,a,nav)==0)//creemos que el nav es inecesario
		nav=nav->dato.sig;
	return desti(nav);
}
	
	
int compara(str q , str a , tData tup){
	int b1,b2; str p,b;
	p=tup->dato.data->dato.data->dato.cad;
	b=tup->dato.data->dato.sig->dato.data->dato.cad;
	b1=equal(p,q);
	b2=equal(a,b);
	if(b1==1 && b2==1){
		return 1;
	}
	else{ 
		return 0;
	}
}
	
str desti(tData tup){
	tData nav=tup->dato.data;
	while(nav->dato.sig != NULL) nav=nav->dato.sig;
	return nav->dato.data->dato.cad;
}	
	
tData deltaNoDet (str q,str a , tData deltita){
	tData nav=deltita->dato.data;
	while(compara(q,a,nav)==0){
		nav= nav->dato.sig;
	}
	return destinodet(nav);
}	
	
	
tData destinodet(tData tup){ //retorna estado destino 
	tData nav = tup->dato.data;
	while(nav->dato.sig!=NULL){
		nav=nav->dato.sig;
	}
	return nav;
}	
	
tData deltaExtendidoNoDet(str q,str w,tData delta){
	tData estado= Inicializa(3);
	
	if(longi(w)==0){
/*		printf("\n se termino la cadena");*/
		estado->dato.data= Inicializa(3);
		estado->dato.data->dato.data= Inicializa(1);
		estado->dato.data->dato.data->dato.cad= q ;
		return estado;
	} 
	else {
		str v=subcad(w) , a=ultisimbolo(w) , pi= NULL; //pi es un estado de P

		tData acum=Inicializa(3), P=deltaExtendidoNoDet(q,v,delta); //P es el conjunto de estados que devuelve una transicion despues de leer v
		tData nav = P->dato.data, var=NULL;
		if (nav!=NULL){
			while(nav!=NULL){
				pi=nav->dato.data->dato.cad;
				var= deltaNoDet(pi ,a,delta);
				acum=Union(acum, var); 
				nav = nav->dato.sig;
			}
			
		}
		else acum=Inicializa(3);
		return acum;
	}
}
	
void cargarAFConParset(TAF* A){
	printf("\n ingrese el conjunto de estados, sin espacios y separados por coma, \n Ej: q0,q1 Q: ");
	A->Q= parset(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	printf("\n ingrese el conjunto de simbolos \n Ej: a,b Alf: ");
	A->Alf= parset(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	printf("\n ingrese el conjunto de transiciones, entre corchetes \n Ej: [q1,a,q2] trans: ");
	A->Trans= parset(  load() ) ; 
	
	printf("\n ingrese el estado inicial qo: ");
	A->EstIni= load() ; //carga una cadena 
	
	printf("\n ingrese el conjunto de estados de Aceptacion F: ");
	A->F= parset(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}
	
void cargarAFPredefinido(TAF* A){
	printf("\n cargando AFD que acepta cadenas sobre 0,1 cuya cantidad de de 0 es impar. ");
	
	A->Q= parset(  load2("q0,q1" ) ) ; //carga una cadena con los estados,  y los convierte en conjunto

	A->Alf= parset(  load2("0,1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Trans= parset(  load2("[q0,0,q1],[q0,1,q0],[q1,0,q0],[q1,1,q1]") ) ; 
	
	A->EstIni= load2("q0") ; //carga una cadena 
	
	A->F= parset(  load2("q1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}	

void cargarAFNDPredefinido(TAF* A){
	
	printf("\n cargando AFD que acepta cadenas sobre 0,1 , que terminan en 1  . ");
	
	A->Q= parset(  load2("q0,q1" ) ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Alf= parset(  load2("0,1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Trans= parset(  load2("[q0,0,{q0}],[q0,1,{q0,q1}],[q1,0,{}],[q1,1,{}]") ) ; 
	
	A->EstIni= load2("q0") ; //carga una cadena 
	
	A->F= parset(  load2("q1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}


