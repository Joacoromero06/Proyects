# include <stdio.h>
# define m 3
# define n 4
void print(const int[m][n]);
int main(){
    int A[m][n]={
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12}
    };
    print(A);
    printf("Size of A:%d\n", sizeof(A));
    return 0;
} 
void print(const int A[m][n]){
    int i, j;
    int cont = 0;
    for (i = 0; i<m; i++)
        for (j = 0; j < n; j++){
            cont++;
            printf("elem %d: %d\n", cont, A[i][j]);
            // Probar asginar: deberia dar error "const" impide
            //A[i][j] = 0;
        }
}
/*
matriz1.c: In function ‘print’:
matriz1.c:22:21: error: assignment of read-only location ‘(*(A + (sizetype)((long unsigned int)i * 16)))[j]’
*/