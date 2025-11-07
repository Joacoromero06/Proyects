%{
#include "ast.h" 
int yylex();
%}

%union{
    struct ast* a;
    double d;
    struct symlist* sl;
    struct symbol* s;
    int fn;
}

%token <d> NUMBER
%token <s> ID
%token <fn> FUNC

%token IF THEN ELSE WHILE DO LET
%token EOL

%nonassoc <fn> CMP 
%right '='
%left '+' '-'
%left '*' '/'
%left UMINUS

%type <a> exp list_exp 
%type <a> stm block
%type <sl> sym_list

%start tree

%%

stm: IF exp THEN block          {$$ = newflow('I', $2, $4, NULL);}
| IF exp THEN block ELSE block  {$$ = newflow('I', $2, $4, $6);}
| WHILE exp DO block            {$$ = newflow('W', $2, $4, NULL);}
| exp
;

block:          {$$ = NULL;}
| stm ';' block {$$ = ($3 == NULL)? $1:newast('L', $1, $3); }
;

exp: exp CMP exp        {$$ = newast($2, $1, $3);} 
| exp '+' exp           {$$ = newast('+', $1, $3);}
| exp '-' exp           {$$ = newast('-', $1, $3);} 
| exp '*' exp           {$$ = newast('*', $1, $3);} 
| exp '/' exp           {$$ = newast('/', $1, $3);} 
| '-' exp %prec UMINUS  {$$ = newast('M', $2, NULL);}
| '(' exp ')'           {$$ = $2;}
| '|' exp '|'           {newast('|', $2, NULL);}
| NUMBER                {$$ = newnum($1);}
| ID                    {$$ = newref($1);}
| ID '=' exp            {$$ = newasgn($1, $3);}
| FUNC '(' list_exp ')' {$$ = newfncall($1, $3);}
| ID '(' list_exp ')'   {$$ = newufncall($1, $3);}
;

list_exp: exp        
| exp ',' list_exp  {$$ = newast('L', $1, $3);}
;

tree: 
| tree stm EOL {printf("=> %f\n", eval($2));freetree($2);}
| tree LET ID '(' sym_list ')' '=' block EOL {dodef($3, $5, $8);printf("defined: %s\n", $3->name);}

sym_list: ID        {$$ = newsymlist($1, NULL);}
| ID ',' sym_list   {$$ = newsymlist($1, $3);}
;

%%