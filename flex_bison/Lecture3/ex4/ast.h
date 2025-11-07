#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h> 

extern int yylineno;
int yyerror(char*, ...);

struct symbol{
    char* name;
    double value;
    struct ast* fn;
    struct symlist* sl;
};

#define NHASH 9997
extern struct symbol symtab[NHASH];
struct symbol* lookup(char*);

struct symlist{
    struct symbol* s;
    struct symlist* next;
};
struct symlist* newsymlist(struct symbol*, struct symlist*);
void freesymlist(struct symlist*);

/* NODE TYPES

* K => double
* I => if_stm
* W => while_stm
* R => reference in symtab
* = => asign_stm
* bifs => built in functions


*/
enum bifs{
    F_sqrt = 1, F_exp, F_log, F_print
};

struct ast{
    int nodetype; //CMP '+' '-' '*' '/' '|'
    struct ast* l;
    struct ast* r;
};
struct num{
    int nodetype; // K
    double number;
};
struct flow{
    int nodetype; // I o W
    struct ast* cond;
    struct ast* tblock;
    struct ast* fblock;
};
struct ref{
    int nodetype; // R
    struct symbol* s;
};
struct asgn{
    int nodetype; // '='
    struct symbol* s;
    struct ast* v;
};
struct fncall{
    int nodetype; // enumbifs
    struct ast* l;
    enum bifs fntype;
};
struct ufncall{
    int nodetype;
    struct ast* l; // parameters == exp_list
    struct symbol* s;
};

// Build AST
struct ast* newast(int, struct ast*, struct ast*);
struct ast* newcmp(int, struct ast*, struct ast*);
struct ast* newnum(double);
struct ast* newflow(int, struct ast*, struct ast*, struct ast*);
struct ast* newasgn(struct symbol*, struct ast*);
struct ast* newfncall(int, struct ast*);
struct ast* newufncall(struct symbol*, struct ast*);
struct ast* newref(struct symbol*);

void dodef(struct symbol*, struct symlist*, struct ast*);

double eval(struct ast*);
void freetree(struct ast*);




