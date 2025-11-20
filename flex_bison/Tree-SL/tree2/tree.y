%{
#include "ast.h"
#include "data.h"
int yyerror(char*, ...);
extern int yylineno;
extern FILE* yyin;
int yylex();
%}
%union{
    tData data;
    struct ast* a;
}
%token <data> STRING_LIT

%type <s> exp




%%
 
block:  exp {printf("=>");Muestra(eval($1));}
;


exp: STRING_LIT         
; 
 
%%
int yyerror(char* s, ...) {
    va_list ap;
    va_start(ap, s);
    fprintf(stderr, "(error %d) ", yylineno);
    vfprintf(stderr, s, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
int main(int argc, char** argv) {
    if(argc == 2){
        FILE* f = fopen(argv[1], "r");
        if(!f){
            yyerror("error el nombre del archivo no existe");
            return 1;
        }
        yyin = f;
    }
    printf("\n=================TREE-SL=================\n");

    yyparse();
    if(yyin != stdin)
        fclose(yyin);
    return 0;
}
