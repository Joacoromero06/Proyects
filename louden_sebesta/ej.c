#include <stdio.h>
int contar();
int gcd(int, int);

int static b = 1;

int main(){
    static int i = 0;
    int d = 0;
//    int* i = &d; error
    for ( int i = 0; i < 10; i++){
        printf("%d\n", contar());
    }
    //printf("gcd(10, 3): %d\n", gcd(10, 3));
    printf("%d\n", i);
    
    while( i < 10){
        static int i;
        i ++;
        d ++;
    //    printf("%d\n", i);
    }
    return 0;
}
int gcd(int a, int b){
    if( a == 0 ) return b;
    return gcd(a-b, a%b);
}
int contar(){
    if( b ) {
        int static x = 0;
        b = 1;
        return x;
    }
    else{
        static int x;
        return x ++;
    } 

}