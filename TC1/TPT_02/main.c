#include "TAD_AF.h"

int main (){
	int opcion ;	
	TAF A;
	
	printf("ingrese 1 si desea cargar un AF por consola en cadena , 2 si desea cargar un AF guiado, ");
	printf("\n 3 si desea usar un AFD predefinido, 4 si desea usar un AFND predefinido  : ");
	scanf("%d", &opcion);
	
	switch( opcion){
		case 1:
			cargarAFConParset(&A);
			break; 
		case 2:
			cargarAF(&A);
			break;
		case 3:
			cargarAFPredefinido(&A);
			break;
			
		case 4:
			cargarAFNDPredefinido(&A);
			break;
			
	}
	mostrarAF(A);
	determinarAF(A);
	
	printf("\n ingrese 1 si desea cargar una cadena y cualquier otro si no lo desea: ");
	scanf("%d", &opcion);
	while(opcion == 1){ 
		determinarCadenaAceptada(A);
		printf("\n opcion: ");
		scanf("%d", &opcion);
	}
	
	return 0;	
}

	
