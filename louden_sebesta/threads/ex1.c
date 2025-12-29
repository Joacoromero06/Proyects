#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main()
{
    printf("ex1: %d\n", getpid());
    char* args[] = {"I am", "going 2 executing", "ex2", NULL};
    execl("./ex2", *args);
    printf("we are back in ex1\n");
    return 0;
}