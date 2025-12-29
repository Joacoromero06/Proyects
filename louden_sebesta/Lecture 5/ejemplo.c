#include <stdio.h>
#include <stdlib.h>
// Solucion: usar variable global, o parametros
int sum = 0;

void printcomputation();

void compute()
{
    // static int sum = 0;
    /*
     * static vuelve sum una variable con scope local a la funcion
     */
    // int sum = 0;
    /*
     * extern no funciona para acceder a referencias locales de otras
     * funciones, solo usado para DECLARAR existencia de variables
     * o funciones en otros archivos
     */

    int n = 10;
    int *values = malloc(sizeof(int) * n);
    for (int *v_i = values; v_i < values + n; v_i++)
    {
        *v_i = 1;
    }

    for (int *v_i = values; v_i < values + n; v_i++)
    {
        sum += (*v_i);
    }
    free(values);

    printcomputation();
}

void printcomputation()
{
    printf("I got the value of sum: %d\n", sum);
}

int main()
{
    compute();
}