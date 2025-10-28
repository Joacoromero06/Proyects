%{
#include <stdio.h>

int yylex(void);
void yyerror(char*);
%}

/* Tokens Declarations */
%token  NUMERO
%token  SUMA RESTA PROD DIV ABS
%token  EOL

%%
listexpr:                   /*epsilon*/       
        | listexpr expr EOL {printf("= %d\n", $2);}
        ;

expr:   expr SUMA term      {$$ = $1 + $3;}
        | expr RESTA term   {$$ = $1 - $3;}
        | term              {$$ = $1;}     /* ADDED | and action */
        ;

term:   term PROD factor    {$$ = $1 * $3;}
        | term DIV factor   {$$ = $1 / $3;}
        | factor            {$$ = $1;}   /* ADDED | */
        ;

factor: NUMERO              {$$ = $1;}     /* ADDED action */
        | ABS expr          {$$ = $2 >= 0? $2 : -$2;}
        ;

%%

int main(){
    yyparse();
    return 0;
}

void yyerror(char* s){
    fprintf(stderr, "Error: %s\n", s);
}