%{
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
int yyerror(char*, ...);
extern int yylineno;
extern FILE* yyin;
int yylex();
%}

%token ID NUMBER BOOL_LIT 
%token IF THEN ENDIF ELSE WHILE FORALL FORANY DO END IN 
%token LET FLECHA
%token FN ENDFN MAIN ENDMAIN RETURN
%token INT SET LIST STRING BOOLEAN

%left UNION 
%left INTER
%left DIFF
%left CONCAT TAKE 
%left ADD KICK
%left OR
%left AND
%right NOT
%left CMP
%left '+' '-'
%left '*' '/'
%right EXPONENCIACION
%nonassoc NEGATIVO

%start tree-sl

%%

tree-sl: header main defs 
;

header: 
        | header fn_prototype
;

main: MAIN ':' block ENDMAIN
;

defs: 
        | defs fn_definition
;

block:  
        | list_stm
;

list_stm:   stm 
        | list_stm stm 
;

stm:    assign_s ';'
        | assign_m ';'
        | while_stm
        | if_stm
        | forall_stm
        | forany_stm
        | return_stm
;

assign_s: ID '=' exp
;

assign_m: LET ID FLECHA '(' list_id ')'
;

while_stm: WHILE '(' exp ')' DO block END
;

if_stm: IF '(' exp ')' block ENDIF
        | IF '(' exp ')' block ELSE block ENDIF
;

forall_stm: FORALL '(' ID IN exp ')' DO block END
;

forany_stm: FORANY '(' ID IN exp '|' exp ')' DO block END
;

exp: ID
        | exp OR exp
        | exp AND exp
        | NOT exp
        | exp CMP exp
        | exp UNION exp
        | exp INTER exp
        | exp DIFF exp
        | exp TAKE exp
        | exp CONCAT exp
        | exp ADD exp
        | exp KICK exp
        | exp '+' exp
        | exp '-' exp
        | exp '*' exp
        | exp '/' exp
        | '|' exp '|'
        | '-' exp %prec NEGATIVO
        | ID '[' exp ']'
        | NUMBER
        | BOOL_LIT
        | lit_struct
        | fn_call
        | '(' exp ')'
;

return_stm: RETURN exp ';'
;

fn_prototype: FN ID '(' list_id ')' ':' type ';'
;

fn_definition: FN ID '(' list_id ')' ':' block ENDFN
;

list_id: ID 
        | list_id ',' ID
;

lit_struct: '{' '}' | '[' ']' 
        | '{' list_exp '}' | '[' list_exp ']' 
;

list_exp: exp 
        | list_exp ',' exp
;

fn_call: ID '(' list_id ')' 
        | ID '(' ')' 
;

type: INT | SET | LIST | STRING | BOOLEAN
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
    printf("\n=================PARCIAL1 TCII=================\n");
    printf("\nNota:\nRecomendamos ingresar el programa en input.txt.\nPara ejecutar ejecute el comando: './tree input.txt' en linux\n\n");

    yyparse();
    if(yyin != stdin)
        fclose(yyin);
    return 0;
}
