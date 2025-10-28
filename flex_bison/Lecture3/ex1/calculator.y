%{
#include <stdio.h>

int yylex();
int yyerror(char*);
%}

%token SUMA RESTA PRODUCTO DIVISION
%token ABS L_PAR R_PAR EOL
%token NUMERO 

%left SUMA RESTA
%left PRODUCTO DIVISION
%right ABS

%%

expression_list: expression_list expression EOL{printf("= %d\n", $2);}
                |
                ;                
expression: term {$$ = $1;}
            | expression SUMA term {$$ = $1 + $3;}
            | expression RESTA term {$$ = $1 - $3;}
            ;
term: factor {$$ = $1;}
    | term PRODUCTO factor {$$ = $1 * $3;}
    | term DIVISION factor {$$ = $1 / $3;}
    ;
factor: NUMERO {$$ = $1;}
        |ABS expression ABS{$$ = $2>=0? $2:-$2;}
        |L_PAR expression R_PAR {$$ = $2;}
        ;

%%

int main(){
    if (yyparse())
    return 0;
}

int yyerror(char* msje){
    fprintf(stderr, "error: %s\n", msje);
    return 0;
}