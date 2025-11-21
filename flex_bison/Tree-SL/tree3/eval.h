#ifndef EVAL_H
#define EVAL_H 

# include "ast.h"

tData eval(struct ast*);
tData evaldata(struct data*);
tData evallist(struct data*);
tData evalopslist(struct ast*);
//tData evalflow(struct flow*);

void treefree(struct ast*);

#endif
