#include <stdio.h>
int main(){
    //ok
    int b = 2, d = 4, c, a;
    a = b + (c = d / b) - 1;
    printf("Test: %d\n", a);
    
    //ok
    --a;    
    printf("Test: %d\n", a);
    
    
    return 0;

}
