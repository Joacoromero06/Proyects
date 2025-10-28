/*for understand should read notebook summary*/
extern int yylineno;
int yyerror(char*, ...);

struct ast {
    int nodetype;
    struct ast* l; 
    struct ast* r; 
};
struct numval {
    int nodetype;
    double number;
};

struct ast* newast(int, struct  ast*, struct ast*);
struct ast* newnumval(double);

double eval(struct ast*);
void freetree(struct ast*);


