#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#define ANCHO 30
#define ALTO 30

//PANTALLA
typedef char Tpantalla[ALTO][ANCHO];
typedef struct {
    float x,y;
}punto;
typedef punto Tcuadrado[4];
void LimpiaPantalla(Tpantalla);
void MuestraPantalla(Tpantalla);
void DibujaCuadrado(Tpantalla,Tcuadrado);
void DibujaLinea(Tpantalla,punto,punto);
int CoordX_IndMat(float);
int CoordY_IndMat(float);

void RotaCuadrado(Tcuadrado,float);
void CreaCuadrado(Tcuadrado);
void RotaVector(punto*,float);

//MAIN
int main(void){
    Tpantalla miframe;
    Tcuadrado micuadrado;
    CreaCuadrado(micuadrado);
    float angulo=0;
    int c=0;
    while(c<15){
        LimpiaPantalla(miframe);
        DibujaCuadrado(miframe,micuadrado);
        angulo+=2;
        RotaCuadrado(micuadrado,angulo);
        MuestraPantalla(miframe);
        usleep(1000000); // 
        system("cls"); // o system("cls") en Windows
        c++;
    }
    MuestraPantalla(miframe);
    

    return 0;
}

void LimpiaPantalla(Tpantalla frame){
    int i,j;
    for(i=0;i<ALTO;i++)
        for(j=0;j<ANCHO;j++)
            frame[i][j]=' ';
}
void MuestraPantalla(Tpantalla frame){
    int i,j;
    for(i=0;i<ALTO;i++){
        for(j=0;j<ANCHO;j++)
            printf("%c",frame[i][j]);
        printf("\n");
    }        
}
void DibujaCuadrado(Tpantalla frame,Tcuadrado cuadrado){
    int i=0;
    for(i;i<=3;i++){
        DibujaLinea(frame,cuadrado[i],cuadrado[(i+1)%4]);
        frame[CoordY_IndMat(cuadrado[i].y)][CoordX_IndMat(cuadrado[i].x)]='A';
    }
}
void DibujaLinea(Tpantalla frame,punto p0,punto p1) {
    int x0 = CoordX_IndMat(p0.x);
    int y0 = CoordY_IndMat(p0.y);
    int x1 = CoordX_IndMat(p1.x);
    int y1 = CoordY_IndMat(p1.y);

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int err = 0,pasox=1,pasoy=1;
    if(x1-x0<0){pasox=-1;}
    if(y1-y0<0){pasoy=-1;}

    if (dx > dy) {
        while (x0 != x1) {
            frame[y0][x0] = '?';
            err += dy;
            if (err >= dx) {
                y0 += pasoy;
                err -= dx;
            }
            x0 += pasox;
        }
    }else {
        while (y0 != y1) {
            frame[y0][x0] = '?';
            err += dx;
            if (err >= dy) {
                x0 += pasox;
                err -= dy;
            }
            y0 += pasoy;
        }
    }
    frame[y0][x0] = '*'; // último punto
}


int CoordX_IndMat(float x){
    return (int)round(x + ANCHO / 2);
}
int CoordY_IndMat(float y){
    return (int)round(y + ALTO / 2);
}


void CreaCuadrado(Tcuadrado cuadrado){
    punto vert1,vert2,vert3,vert4;
    
    vert1.x=10;vert1.y=10;
    vert2.x=-10;vert2.y=10;
    vert3.x=-10;vert3.y=-10;
    vert4.x=10;vert4.y=-10;

    cuadrado[0]=vert1;
    cuadrado[1]=vert2;
    cuadrado[2]=vert3;
    cuadrado[3]=vert4;
}
void RotaCuadrado(Tcuadrado cuadrado,float angulo){
    RotaVector(&cuadrado[0],angulo);
    RotaVector(&cuadrado[1],angulo);
    RotaVector(&cuadrado[2],angulo);
    RotaVector(&cuadrado[3],angulo);
}
void RotaVector(punto* vec, float angulo){
    float x = vec->x;
    float y = vec->y;
    vec->x = x * cos(angulo) - y * sin(angulo);
    vec->y = x * sin(angulo) + y * cos(angulo);
}
