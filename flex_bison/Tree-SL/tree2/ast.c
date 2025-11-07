# include "ast.h"

struct ast* newast(enum nodetype tipo, struct ast* l, struct ast* r){
    struct ast* a = malloc(sizeof(struct ast));
    if(!a){printf("Sin Memoria en newast\n"); return NULL;}
    a->nodetype = tipo;
    a->l = l;
    a->r = r;
    return a;
}

struct ast* newdata(enum datatype tipo, tData data){
    struct data* a = malloc(sizeof(struct data));
    if(!a){printf("Sin Memoria en newast\n"); return NULL;}
    a->datatype = tipo;
    a->data = data;
    return (struct ast*) a;
}

tData eval(struct ast* a){
    tData v = NULL;
    if(!a){printf("Sin Memoria en newast\n"); return NULL;}
    switch (a->nodetype){
    case DATA:
        struct data* a_ = (struct data*)a; 
        switch ( a_-> datatype ){    
        case STR: v = a_->data; break;
        
        default: break;
        }
    break;
    default: break;
    }
    return v;
}

void freetree(struct ast*){}
