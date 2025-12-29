#include <stdio.h>

int adder(int*, int);

int main()
{
    int array[] = {1,2,3};
    int tam = 3;
    printf("the sum is: %d\n", adder(array, tam));
    printf("the sum is: %d\n", adder(array, tam));
    printf("the sum is: %d\n", adder(array, tam));


    return 0;
}
int adder(int* arr, int n)
{
    static int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    return sum;
}