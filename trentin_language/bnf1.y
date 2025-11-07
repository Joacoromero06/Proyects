extern int yylineno; /* from lexer */
void yyerror(char *s, ...);
/* nodes in the abstract syntax tree */
struct ast {
int nodetype;
struct ast *l;
struct ast *r;
};
struct numval {
int nodetype;
double number;
};

struct ast *newast(int nodetype, struct ast *l, struct ast *r);
struct ast *newnum(double d);
/* evaluate an AST */
double eval(struct ast *);
/* delete and free an AST */
void treefree(struct ast *);

%{
# include <stdio.h>
# include <stdlib.h>
# include "fb3-1.h"
%}
%union {
struct ast *a;
double d;
}
/* declare tokens */
%token <d> NUMBER
%token EOL
%type <a> exp factor term

%%
calclist: /* nothing */
| calclist exp EOL {
printf("= %4.4g\n", eval($2));
treefree($2);
printf("> ");
}
| calclist EOL { printf("> "); } /* blank line or a comment */
;

declaration: //??
asign: ID IGUAL expr { $1 = $3; }//hay que hacer newast?
asign_multiple: ID FLECHA IGUAL L_PARENTESIS id_list R_PARENTESIS
{ asigna($1 = $5) }

exp_arit: factor_arit
| exp_arit '+' factor_arit { $$ = newast('+', $1,$3); }
| exp_arit '-' factor_arit { $$ = newast('-', $1,$3);}
;
factor_arit: term_arit
| factor_arit '*' term_arit { $$ = newast('*', $1,$3); }
| factor_arit '/' term_arit { $$ = newast('/', $1,$3); }
;
term_arit: NUMBER { $$ = newnum($1); }
| '|' term_arit '|' { $$ = newast('|', $2, NULL); }
| '(' exp_arit ')' { $$ = $2; }
| '-' term_arit { $$ = newast('M', $2, NULL); }
;

//EXPRESIONES STRUCT -> evaluan a struct
exp_struct: factor_struct {$$ = $1;}
| exp_struct (UNION|INTERSECCION|DIFERENCIA) factor_struct {$$ = union($1, $3)}
factor_struct: elem {$$ = $1;}
| elem (ADD|KICK) factor_struct { inserta($1, $3);}

elem: atom
| struct_list
| struct_set

atom: ID
|STRING_LIT
|INT_LIT
|BOOL_LIT

struct_list: L_CORCHETE [elem_list] R_CORCHETE {$$ = newlist_de($2)}
struct_set:  L_LLAVE [elem_list] R_LLAVE

//cambiar usar mas vars o opcionas de ebnf
elem_list: elem //logica para crear ast vacio y insertar elementos
| elem COMA elem_list

//EXPRESIONES RELACIONALES -> evaluan boolean
expr_rel: expr OP_REL  {}
expr_log: [NOT] expr_rel OP_LOG expr_log
| [NOT] expr_rel

//EXPRESION
expr: exp_arit
| exp_struct
| expr_log
| expr_rel

if_emparejado: IF cond THEN block ELSE block ENDIF
if_noemparejado: IF cond THEN block ENDIF

%%