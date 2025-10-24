#include <stdio.h>
typedef int vec_ent [100];
/*Modulo para ingresar un entero*/
int pedirEntero();
/*Modulo que calcula la cantidad de digitos de un nro entero*/
int cantDigitos(int);
/*Prototipo del modulo que determina si un numero es primo o no*/
int esPrimo(int);
/*El procedmiento del ejemplo 1*/
void mi_ej1();
int main() {
    vec_ent v[12];
    int n, acu;
    n = 10;
    acu = 0;
    for (int i = 1; i <= n; i++)
        acu += i;
    printf("la sumatoria desde 1 hasta %d es: %d", n, acu);
    return 0;
}

/*Pide el numero por consola y lo devuelve en su nombre -> es una funcion*/
int pedirEntero(){
    int x;
    printf("\nIngrese un nro entero: ");
    scanf("%d", &x);
    return x;
}
/*Descomponemos el numero y contamos cuantos digitos tiene */
int cantDigitos(int aux){
    int c = 0; /* variable contador de digitos a retornar*/
    while ( aux != 0)
    { 
        c++;
        aux = aux / 10;
    }
    return c;
}
/*Determinamos si el nuimero es primo con el algoritmo clasico de EP*/
int esPrimo(int x){
    int pd, mitad, b;
    b = 0;
    pd = 2;
    mitad = x / 2;
    while (pd <= mitad && x % pd != 0)
        pd++;
    
    if ( pd > mitad && x != 1){
        b = 1;
    }
    return b;      
}
void mi_ej1(){
    int x, c;
    x = pedirEntero();
    c = cantDigitos(x);/*x pasa por valor, su valor no se pierde*/
    
    printf("\nLa cantidad de digitos que tiene el nro: %d es %d\n", x, c);
    if ( esPrimo(x) )
        printf("\nx: %d Si es primo", x);
    else
        printf("\nx: %d No es primo", x);
}


