%{
#include <stdlib.h>
#include <stdio.h>
#include "ast2.h"
//int yylex();
%}
%union{
    struct ast* a;
    double d;
}
%token <d> NUMBER
%token EOL
%type <a> exp


%%
calcist: calcist exp EOL{
    printf("= %f\n", eval($2));
    printf("$$>");
    freetree($2);
}
|calcist EOL{printf("$$>");}
|;
exp: exp '+' exp {$$ = newast('+', $1, $3);}
    |exp '-' exp {$$ = newast('-', $1, $3);}
    |exp '*' exp {$$ = newast('*', $1, $3);}
    |exp '/' exp {$$ = newast('/', $1, $3);}
    |'|' exp '|'{$$ = newast('|', $2, NULL);}
    |'(' exp ')'{$$ = newast('(', $2, NULL);}
    | '-' exp {$$ = newast('M', $2, NULL);}
    | NUMBER {$$ = newnumval($1);}
%%