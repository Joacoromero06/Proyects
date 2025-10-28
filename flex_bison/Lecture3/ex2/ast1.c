#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "ast1.h"
#include "calculator1.tab.h"

struct ast* newast(int nodetype, struct ast* l, struct ast* r){
    struct ast* a = malloc(sizeof(struct ast));
    if(!a){yyerror("sin memoria ast"); exit(1);}
    a->nodetype = nodetype;
    a->l = l;
    a->r = r;
    return a;
}
struct ast* newnumval(double d){
    struct numval* a = malloc(sizeof(struct numval));
    if(!a){yyerror("sin memoria numval"); exit(1);}
    a->nodetype = 'K';
    a->number = d;
    return (struct ast*) a;
}
double eval(struct ast* a){
    double v;
    switch (a->nodetype){
    case 'K': v = ((struct numval*)a)->number; break;
    case '+': v = eval(a->l)+eval(a->r); break;
    case '-': v = eval(a->l)-eval(a->r); break;
    case '*': v = eval(a->l)*eval(a->r); break;
    case '/': v = eval(a->l)/eval(a->r); break;
    case '|': v = eval(a->l); if (v<0) v = -v; break;
    case '(': v = eval(a->l); break;
    case 'M': v = -eval(a->l); break;
    
    default: yyerror("nodetype indefinido: %c", a->nodetype); break;
    }
    return v;
}
void freetree(struct ast* a){
    switch (a->nodetype){
    case '+': case '-': case '*': case '/': 
        freetree(a->r); 
    case '|': case '(': case 'M': 
        freetree(a->l);
    case 'K': free(a); break;
    default: yyerror("nodetype indefinido: %c", a->nodetype); break;
    }/*switch sin break, genera que desde el case donde se cumple se ejecuta todo lo de abajo*/
}
int yyerror(char* s, ...){
    va_list ap; /*va_list: tipo definido en stdarg, puntero a lista de argumentos*/
    va_start(ap, s); /*Inicializa el puntero al siguiente argumento despues de s*/
    
    fprintf(stderr, "(error %d)", yylineno);
    vfprintf(stderr, s, ap);
    /*
    en el stream stderr, se agrega la cadena de formato s, los valores de los
    formatos estan en ap. 
    */
    fprintf(stderr, "\n");
    va_end(ap);/*free de va_list ap*/
}
int main(){
    yyparse();
    return 0;
}

