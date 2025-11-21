# include "error.h"
# include "defs.h"

void yyerror(const char* s){
    fprintf(stderr, "Syntax error %s\n", s);
}