#ifndef AST_H
#define AST_H


# include "data.h"

struct ast{
    int nodetype;
    struct ast* l;
    struct ast* r;
};
struct data{
    int nodetype;
    struct ast* l; // LIST_OF_AST
    tData d;
};
/*
struct flow{
    int nodetype;
    struct symbol* s; 
    struct ast* l;
    struct ast* r;
    struct ast* cond;
};*/


struct ast* newast(int, struct ast*, struct ast*);
struct ast* newdata(int, struct ast*, tData);



#endif
