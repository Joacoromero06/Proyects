#include "../Assets/af.h"

//Procedimientos para Entrada/Salida AF
void CargarAF( TAF* A){
	int N, opcion;
	printf("\n ingrese el conjunto de estados del automata Q: ");
	A->Q=  Inicializa(3);
	printf("\n cuantos estados desea cargar: ");
	scanf("%d", &N);
	CargaConj(A->Q, N);   
	
	printf("\n ingrese el conjunto de simbolos, (alfabeto): ");
	A->Alf= Inicializa(3);
	printf("\n cuantos simbolos desea cargar: ");
	scanf("%d", &N);
	CargaConj(A->Alf, N); 
	
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
	CargaConj(A->F, N);    
}
void MostrarAF (TAF A){
	printf("\nMOSTRANDO AUTOMATA");
	if(DeterminarAFD(A.Trans)==-1){printf(" DETERMINISTA\n");}
	else{printf(" NO DETERMINISTA\n");}
	printf("\nCONJUNTO DE ESTADOS: ");Muestra(A.Q);   
	
	printf("\nALFABETO: ");Muestra(A.Alf);  
	
	printf("\nFUNCION DELTA: ");Muestra(A.Trans);  
	
	printf("\nESTADO INICIAL: ");print(A.EstIni);  
	
	printf("\nCONJUNTO DE ESTADOS DE ACEPTACION: ");Muestra(A.F);
}
void CargaConj(tData Q, int N){
	int i;
	for(i=1; i<= N; i++ ){
		printf("\nIngrese:");
		AgregaCadena(Q);
	}
}	

//Funciones y Procedimientos para AFD
void CargaDeltaDet(tData delta, int N){
	int i;
	tData tupla= NULL;
	
	for(i=1; i<= N; i++ ){
		tupla= CargaTuplaDet();
		AgregaXcolaData(delta, tupla);
	}	
}
tData CargaTuplaDet(){
	
	tData tupla= Inicializa(2);
	printf("\n ingrese estado actual: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese simbolo: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese estado destino: ");
	AgregaCadena(tupla);
	
	return tupla;
}
tData CargaTuplaDet2(str q,str a,str dest){
	tData tupla= Inicializa(2);
	AgregaXcolaCad(tupla,q);AgregaXcolaCad(tupla,a);AgregaXcolaCad(tupla,dest);
	return tupla;
}
str DeltaExtendidoDet(str q,str w,tData delta){
	if(longi(w)==0){
		return q;
	}
	else{
		str v,a;
		v=subcad(w);
		a=ultisimbolo(w);
		return DeltaDet(DeltaExtendidoDet(q,v,delta),a,delta);
	}
}		
str DeltaDet (str q, str a,tData delta){
	tData nav=delta->dato.data;
	while (nav!=NULL && Compara(q,a,nav)==0)
		nav=nav->dato.sig;
	if(nav==NULL){return NULL;}
	return EstDest(nav);
}
str EstDest(tData tup){
	tData nav=tup->dato.data;
	while(nav->dato.sig != NULL) nav=nav->dato.sig;
	return nav->dato.data->dato.cad;
}

// Funciones para AFND 
void CargaDeltaNoDet(tData delta, int N){
	int i;
	tData tupla= NULL;
	
	for(i=1; i<= N; i++ ){
		tupla= CargaTuplaNoDet();
		AgregaXcolaData(delta, tupla);
	}	
}
tData CargaTuplaNoDet(){
	
	tData tupla= Inicializa(2) , conjDest= Inicializa(3) ;
	int N;
	
	printf("\n ingrese estado actual: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese simbolo: ");
	AgregaCadena(tupla);
	
	printf("\n ingrese la cantidad de estados destino: ");
	scanf("%d", &N);
	CargaConj(conjDest, N);
	AgregaXcolaData(tupla , conjDest);
	
	return tupla;
}
tData DeltaExtendidoNoDet(str q,str w,tData delta){
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
		tData acum=Inicializa(3), P=DeltaExtendidoNoDet(q,v,delta); //P es el conjunto de estados que termina el AFN despues de leer v
		tData nav = P->dato.data, rtdo=NULL;
		if (nav!=NULL)
			while(nav!=NULL){
				pi=nav->dato.data->dato.cad;
				rtdo= DeltaNoDet(pi ,a,delta);
				acum=Union(acum, rtdo); 
				nav = nav->dato.sig;
			}	
		else 
			acum=Inicializa(3);
		return acum;
	}
}
tData DeltaNoDet (str q,str a , tData deltita){
	tData nav=deltita->dato.data;
	while(Compara(q,a,nav)==0){
		nav= nav->dato.sig;
	}
	return ConjDest(nav);
}			
tData ConjDest(tData tup){ //retorna estado destino 
	tData nav = tup->dato.data;
	while(nav->dato.sig!=NULL){
		nav=nav->dato.sig;
	}
	return nav;
}	

//afnd->afd
TAF afnd2Afd (TAF N){
	TAF D; 				 
	D.EstIni=Data2Str(parsetPlus(N.EstIni));		printf("\nEl estado inicial es: ");print(D.EstIni);
	D.Q=getEstruct_str(D.EstIni,3);					printf("\nEl conjunto de estados de D al inicio es: ");Muestra(D.Q);
	D.Alf=N.Alf;D.Trans=Inicializa(3);
	
	str S_cad=NULL;/*"{q0,q1}"*/tData S_conj=NULL/*{q0,q1}*/,navS=NULL;
	str qNuevo_cad=NULL;		tData qNuevo_conj=NULL;
	
	str a=NULL,p=NULL/*"q0"*/;	tData navAlf=NULL, elemAux=NULL,transicionNueva=NULL;
	
	S_cad=buscaQnoDef(D.Q,D.Trans,D.Alf);  
	while (S_cad!=NULL){							printf("\nEl estado de D que no fue definido es: ");print(S_cad);
		S_conj=parset(S_cad);						printf("\nRepresenta el conjunto de estados simultaneos de N: ");Muestra(S_conj);
		navAlf=(D.Alf)->dato.data;
		while (navAlf!=NULL){
			a=getCadena(navAlf);					printf("\n\nVAMOS A DEFINIR LA TRANSICION DEL ESTADO CON EL SIMBOLO: ");print(a);
			navS=S_conj->dato.data;					printf("\nPara los estados de N: ");print(Data2Str(navS));
			qNuevo_conj=Inicializa(3);
			while (navS!=NULL){ 
				p=getCadena(navS);					printf("\nPara el estado: ");print(p);
				qNuevo_conj=Union(qNuevo_conj,DeltaNoDet(p,a,N.Trans));
				avanzo(&navS);						printf(". N pasa al conjunto de estados: ");Muestra(DeltaNoDet(p,a,N.Trans));
			}
			Ordena(qNuevo_conj);					printf("\n\nEl conj de estados al que pasa N es: ");Muestra(qNuevo_conj);//{q1,q2}
			qNuevo_cad=Data2Str(qNuevo_conj);		printf("  ->Corresponde un unico estado de D: ");print(qNuevo_cad);//"{q1,q2}"
			transicionNueva=CargaTuplaDet2(S_cad,a,qNuevo_cad);printf("\nEntonces armamos la trancision: ");Muestra(transicionNueva);
	
			D.Trans=Union(D.Trans,getEstruct_data(transicionNueva,3));printf("\nEl conj de transiciones actualizado es: ");Muestra(D.Trans);
			D.Q=Union(D.Q,getEstruct_str(qNuevo_cad,3));printf("\nEl conj estados actualizado es: ");Muestra(D.Q);
			avanzo(&navAlf);
		}
		S_cad=buscaQnoDef(D.Q,D.Trans,D.Alf);
	}
	D.F=Inicializa(3); 
	tData navQ=D.Q->dato.data,conjAux=NULL;
	while (navQ!=NULL){
		S_conj=parset(getCadena(navQ));
		if(Cardinalidad(Interseccion(S_conj,N.F))>0){
			conjAux=getEstruct_str(getCadena(navQ),3);
			D.F= Union(D.F,conjAux);
		}
		avanzo(&navQ);	
	}
	return D;
}
str buscaQnoDef(tData Q, tData delta, tData alf){	printf("\n\nVAMOS A BUSCAR UN ESTADO EN D CON TRANSICION NO DEFINIDA:");
	tData navQ=Q->dato.data, navAlf=NULL;
	str q=NULL,a=NULL,destino;
	printf("\nEl Conjunto donde buscamos es: ");Muestra(Q);
	while (navQ!=NULL){
		q=getCadena(navQ);
		navAlf=alf->dato.data;
		while (navAlf!=NULL){
			printf("\nPara el estado: ");print(q);
			a=getCadena(navAlf);printf("y el simbolo : ");print(a); 
			destino=DeltaDet(q,a,delta);printf("-> destino es: ");print(destino);
			if(destino==NULL){return q;}
			avanzo(&navAlf);
		}
		avanzo(&navQ);
	}
	printf("\nTODOS LOS ESTADOS EN D TIENEN TRANSICION DEFINIDA\n");
}

void ordenaEstados(TAF* A){//¿mejorar?
	Ordena(A->Alf);Ordena(A->Q);Ordena(A->F);Ordena(A->Trans);
}
void renombrarEstados(TAF* D){
	if(DeterminarAFD(D->Trans)!=-1){return;}
	int n=Cardinalidad(D->Q),i;
	str qi=NULL,qCambiar;
	tData navQ=D->Q->dato.data;
	for(i=1;i<=n;i++){
		qi=armaCad("q",i);
		qCambiar=getCadena(navQ);
		setCadena(navQ,qi);
		renombrarDelta(D->Trans,qCambiar,qi);
		renombrarF(D->F,qCambiar,qi);
		D->EstIni= load2("q1"); //mejor seria una bandera que haga q qi sea el estado inicial solo cuando es la 1ra iteracion del for
		avanzo(&navQ);
	}	
}
void renombrarDelta(tData delta,str qCambiar,str qi){
	tData nav=delta->dato.data;
	while(nav!=NULL){
		cambiaTupla(nav,qCambiar,qi);
		avanzo(&nav);
	}
}
void cambiaTupla(tData tupla,str qCambiar,str qi){
	tData nav=tupla->dato.data;
	while(nav!=NULL){
		if(equal(qCambiar,getCadena(nav)))
			setCadena(nav,qi);
		avanzo(&nav);
	}
}
void renombrarF(tData F,str qCambiar,str qi){
	tData nav=F->dato.data;
	while (nav!=NULL){
		if(equal(qCambiar,getCadena(nav)))
			setCadena(nav,qi);
		avanzo(&nav);
	}
}
void afndToAfd(TAF N){
	MostrarAF(N);
	if(DeterminarAFD(N.Trans)!=-1){
		TAF D=afnd2Afd(N);
		renombrarEstados(&D);
		MostrarAF(D);
	}else{
		printf("\nTiene que ser no determinsta. ERROR.");
	}	
}

//Funciones y Procedimientos para modelar AF
int DeterminarAFD(tData delta){//-1 si es afd
	tData nav= NULL, nav2= NULL; 
	if(delta != NULL){
		nav= delta->dato.data;//accedo a una tupla
		if(nav != NULL){
			nav2= nav->dato.data;
			while( nav2->dato.sig != NULL ){//voy hasta el final de la tupla
				nav2= nav2->dato.sig ;
			}
			return Cardinalidad(nav2);//-1 si es un estado, mas si es un conj de estados
		}
		return -1; //no pasa
	}
	return -1; //no pasa
}	
int DeterminaCadenaAceptada(TAF A,str w){//0 no es aceptada
	str q=NULL;
	tData P= NULL;
	tData estFinal= Inicializa(3) , interseccionDeEstados= NULL ;
	
	if(DeterminarAFD( A.Trans ) == -1 ){ //es determinista
		q= DeltaExtendidoDet(A.EstIni, w, A.Trans);
		estFinal->dato.data= Inicializa(1);
		estFinal->dato.data->dato.cad= q ;
		return Pertenece(estFinal,A.F);
	}
	else{
		P = DeltaExtendidoNoDet(A.EstIni, w, A.Trans);
		interseccionDeEstados= Interseccion( P , A.F );
		return Cardinalidad(interseccionDeEstados);	
	}
	
	
}
void CargarParset(TAF* A){//chequear si se puede cargar afnd
	printf("\n ingrese el conjunto de estados, sin espacios y separados por coma, \n Ej: q0,q1 Q: ");
	getchar();
	A->Q= parsetPlus(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	printf("\n ingrese el conjunto de simbolos \n Ej: a,b Alf: ");
	A->Alf= parsetPlus(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	printf("\n ingrese el conjunto de transiciones, entre corchetes \n Ej: [q1,a,q2] trans: ");
	A->Trans= parsetPlus(  load() ) ; 
	
	printf("\n ingrese el estado inicial qo: ");
	A->EstIni= load() ; //carga una cadena 
	
	printf("\n ingrese el conjunto de estados de Aceptacion F:\n Ej: q2 F:");
	A->F= parsetPlus(  load() ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}	
void CargaXcadena(TAF* A){
	printf("Escriba su automata como 5-upla\n"); 
	getchar();
	str cadAF=load(),aux=cadAF;
	if(1)//chequear que sea correcta
	cadAF=afterToken(cadAF,'(');
	cadAF=beforetoken(cadAF,')');
	printf("\nCadena despues de sacar los parentesis: ");print(cadAF);
	
	str cadQ=beforetoken(cadAF,'}');AgregaXcola(&cadQ,'}');
	printf("\nCadena del conjunto; ");print(cadQ);

	cadAF=afterToken(cadAF,'}');cadAF=afterToken(cadAF,',');
	
	str cadAlf=beforetoken(cadAF,'}');AgregaXcola(&cadAlf,'}');
	printf("\nCadena del Alfabeto; ");print(cadAlf);

	cadAF=afterToken(cadAF,'}');cadAF=afterToken(cadAF,',');

	str cadDelta=beforetoken(cadAF,'}');AgregaXcola(&cadDelta,'}');
	printf("\nCadena del conjunto de transiciones: ");print(cadDelta);

	cadAF=afterToken(cadAF,'}');cadAF=afterToken(cadAF,',');

	str cad_q0=beforetoken(cadAF,',');cadAF=afterToken(cadAF,',');
	printf("\nCadena del estado inicial: ");print(cad_q0);

	str cadF=beforetoken(cadAF,',');

	A->Q=parset(cadQ);
	A->Alf=parset(cadAlf);
	A->Trans=parset(cadDelta);
	A->EstIni=cpy(cad_q0);
	A->F=parset(cadF);
}
int Compara(str q , str a , tData tup){//Funcion chuequea si la tup es la func de trans para q y a
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

//AFs predefinidos
void CargarAFDPredefinido(TAF* A){
	printf("\n cargando AFND que acepta cadenas sobre 0,1 cuya cantidad de de 0 es impar. ");
	
	A->Q= parsetPlus(  load2("q0,q1" ) ) ; //carga una cadena con los estados,  y los convierte en conjunto

	A->Alf= parsetPlus(  load2("0,1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Trans= parsetPlus(  load2("[q0,0,q1],[q0,1,q0],[q1,0,q0],[q1,1,q1]") ) ; 
	
	A->EstIni= load2("q0") ; //carga una cadena 
	
	A->F= parsetPlus(  load2("q1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}	
void CargarAFNDPredefinido(TAF* A){
	
	printf("\n cargando AFD que acepta cadenas sobre 0,1 , que terminan en 1  . ");
	
	A->Q= parsetPlus(  load2("q0,q1" ) ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Alf= parsetPlus(  load2("0,1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
	A->Trans= parsetPlus(  load2("[q0,0,{q0}],[q0,1,{q0,q1}],[q1,0,{}],[q1,1,{}]") ) ; 
	
	A->EstIni= load2("q0") ; //carga una cadena 
	
	A->F= parsetPlus(  load2("q1") ) ; //carga una cadena con los estados,  y los convierte en conjunto
	
}

	
void CargaXcadenaNoDet(TAF* A){
	
	printf("Escriba su automata como 5-upla\n"); 
	getchar();
	str cadAF=load(),aux=cadAF;
	if(1)//chequear que sea correcta
		cadAF=afterToken(cadAF,'(');
	cadAF=beforetoken(cadAF,')');
	printf("\nCadena despues de sacar los parentesis: ");print(cadAF);
	
	str cadQ=beforetoken(cadAF,'}');AgregaXcola(&cadQ,'}');
	printf("\nCadena del conjunto; ");print(cadQ);
	
	cadAF=afterToken(cadAF,'}');cadAF=afterToken(cadAF,',');
	
	str cadAlf=beforetoken(cadAF,'}');AgregaXcola(&cadAlf,'}');
	printf("\nCadena del Alfabeto; ");print(cadAlf);
	
	cadAF=afterToken(cadAF,'}');cadAF=afterToken(cadAF,',');
	
/*	{[q1,1,{q2}],[q1,2,{q2}],[q2,1,{q1}],[q2,1,{}]},q1,{q2})*/
	
	str cadDelta= getConjuntoTrans(&cadAF);
	printf("\nCadena del conjunto de transiciones: ");print(cadDelta);
		
	str cad_q0=beforetoken(cadAF,',');cadAF=afterToken(cadAF,',');
	printf("\nCadena del estado inicial: ");print(cad_q0);
	
	str cadF=beforetoken(cadAF,',');
	
	A->Q=parset(cadQ);
	A->Alf=parset(cadAlf);
	A->Trans=parset(cadDelta);
	A->EstIni=cpy(cad_q0);
	A->F=parset(cadF);
	
}

str getConjuntoTrans(str* cadAF){
	str nav= *cadAF , next= (*cadAF)->sig;
	str cadenaAux= NULL;
	
	while( nav->caracter != ']' || next->caracter != '}'){
		printf("\n el nav es: %c",nav->caracter);
		printf("\t el next es: %c",next->caracter);
		AgregaXcola(&cadenaAux, nav->caracter);
		nav=next;
		next=next->sig;
		printf("\n avanzo");
		printf("\n el nav es: %c",nav->caracter);
		printf("\t el next es: %c",next->caracter);
	}
	AgregaXcola(&cadenaAux, ']');
	AgregaXcola(&cadenaAux, '}');
	
	*cadAF= next->sig->sig; //apuntare al estado inicial 
	return cadenaAux;
}




