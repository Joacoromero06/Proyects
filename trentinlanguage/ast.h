#include "data.h"
#include <stdio.h>
#include <stdlib.h>

struct ast{
    int nodetype;
    struct ast* l;
    struct ast* r;
    tData d;
};

struct ast* newast(int, struct ast*, struct ast*, tData);

int get_nodetype(struct ast*);

tData eval(struct ast*);


int yyerror(char*);