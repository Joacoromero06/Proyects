# include <stdio.h>
# include <stdlib.h>
# include <stdarg.h>
# include "data.h"

enum nodetype{
    LIST_EXP, SYMBOL, UNION, DATA
};

struct ast{
    enum nodetype nodetype;
    struct ast* l;
    struct ast* r;
};

struct data{
    enum nodetype nodetype;
    enum datatype datatype;
    tData data;
}; 

struct ast* newast(enum nodetype, struct ast*, struct ast*);
struct ast* newdata(enum datatype, tData);

tData eval(struct ast*);
void freetree(struct ast*);


