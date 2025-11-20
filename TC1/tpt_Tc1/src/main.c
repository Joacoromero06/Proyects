#include "Assets/af.h"

void MenuInicio(int*);
void MenuInteractivo(TAF);
int main (){
	TAF A;
	int opcion;
	do{
		MenuInicio(&opcion);
		switch( opcion){
			case 1:
				CargarParset(&A);
				break; 
			case 2:
<<<<<<< Updated upstream
				CargaXcadena(&A);
=======
				//CargarAF(&A);
>>>>>>> Stashed changes
				break;
			case 3:
				CargarAFDPredefinido(&A);
				break;
				
			case 4:
				CargarAFNDPredefinido(&A);
				break;	
			case 5:
				afndToAfd(A);
				break;
			case 6:
				CargarParset(&A);
				break;
			case 7:
				CargaXcadenaNoDet(&A);
				break;
		}
		MostrarAF(A);
		MenuInteractivo(A);
	}while(opcion!=0);
	printf("\n\nFIN DEL PROGRAMA\n\n");
	return 0;	
}
void MenuInicio(int* opc){
	printf("\n---------------MENU---------------");
	printf("\n0 SALIR");
	printf("\n1 CARGAR AF POR PARTES");
<<<<<<< Updated upstream
	printf("\n2 CARGAR AF POR CADENA");
=======
	//printf("\n2 CARGAR AF GUIADO");
>>>>>>> Stashed changes
	printf("\n3 USAR AFD PREDEFINIDO");
	printf("\n4 USAR AFND PREDEFINIDO");
	printf("\n5 PASAR AFND A AFD");
	printf("\n6 CARGAR AFND POR PARTES");
	printf("\n7 CARGAR AFND POR CADENA");
	printf("\n: ");
	scanf("%d", opc); 
}
void MenuInteractivo(TAF A){
	int opcion;
	printf("\n1 seguir o 0 salir. Opcion elegida: ");
	scanf("%d", &opcion); 
	while(opcion == 1){
		printf("\nPruebe una cadena: ");
		getchar();
		if(!DeterminaCadenaAceptada(A,load()))
			printf("\nCadena no aceptada");
		else
			printf("\nCadena aceptada");
		printf("\n1 seguir o 0 salir. Opcion elegida: ");
		scanf("%d", &opcion);  
	}
}

/*({q1,q2},{1,2},{[q1,1,{q2}],[q1,2,{q2}],[q2,1,{q1}],[q2,2,{}]},q1,{q2})*/
	
