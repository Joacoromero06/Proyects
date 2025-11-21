# include "ast.h"
# include "defs.h"

 

struct ast* newast(int nodetype, struct ast* l, struct ast* r){
    struct ast* a = malloc(sizeof(struct ast));
    if (!a) { printf("Error en malloc newast\n"); return NULL; }

    a->nodetype = nodetype;
    a->l = l;
    a->r = r;
    
    return a;
}

struct ast* newdata(int nodetype, struct ast* l, tData d){
    struct data* a = malloc(sizeof(struct data));
    if (!a) { printf("Error en mallo newdata\n"); return NULL; }

    a->nodetype = nodetype;
    a->l = l;
    a->d = d;

    return (struct ast*) a;
}

/*
struct flow* newflow(int flowtype, struct symbol*s, struct ast* l, struct ast* r, struct ast* cond){
    struct flow* f = malloc(sizeof(struct flow));
    if(!f) { printf("Error jaja rip pc\n"); return NULL; }

    f->nodetype = flowtype;
    f->l = l;
    f->r = r;
    f->cond = cond;

    return (struct ast*) f;
} */  
