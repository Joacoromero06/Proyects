%{
#include <stdio.h>
#include <stdlib.h>
#include "ast1.h"
int yylex(void);

%}

%union{
    struct ast* a;
    double d;
}
%token <d> NUMBER
%token EOL
%type <a> exp term factor
 
%%

calclist: 
        | calclist exp EOL{
            printf("= %f\n", eval($2));
            freetree($2);
            printf("$$> ");
        }
        | calclist EOL {printf("$$>");}
;
exp: term
    | exp '+' term {$$ = newast('+', $1, $3);}
    | exp '-' term {$$ = newast('-', $1, $3);}
;
term: factor
    | term '*' factor {$$ = newast('*', $1, $3);}
    | term '/' factor {$$ = newast('/', $1, $3);}
;
factor: NUMBER {$$ = newnumval($1);}
    | '|' exp '|'  {$$ = newast('|', $2, NULL);}
    | '(' exp ')' {$$ = newast('(', $2, NULL);}
    | '-' factor {$$ = newast('M', $2, NULL);}
        
%%
