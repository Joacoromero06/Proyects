#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
int main(int nargs, char** argv)
{
    printf("ex2: %d\n", getpid());
    return 0;
}