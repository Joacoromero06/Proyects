# include <stdio.h>
# include <math.h>

/*Calcula el mcd entre dos flotantes*/
double mcd_double(double, double);

/*Ingresa un double y lo retorna*/
double ingreso_flotante();


/*Ingresa un int y lo retorna*/
int ingreso_entero();

int main(){
    //extern int tol;
    double a, b;
    a = ingreso_flotante();
    b = ingreso_flotante();
    //tol = ingreso_entero();
    printf("\nEl mcd entre %2.f y %f es: %f", a, b, mcd_double(a,b));
    
}

double ingreso_flotante(){
    double x;
    printf("\nIngrese un nro:");
    scanf("%f", &x);
    return x;
}

double mcd_double(double a, double b){
    double coc, resto;
    int i;
    int tol = 100;
    i = 0;
    while (i < tol && fabs(b) < 0.0001 ){
        resto = fmod(a,b);
        a = b;
        b = resto;
    }
    return a;
    
}

int ingreso_entero(){
    int x;
    printf("\nIngrese un nro:");
    scanf("%d", &x);
    return x;
}

