%{
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include "defs.h"
    #include "data.h"
    #include "ast.h"  
    
    int yylex();
    int yyerror(char* s);
%}
%union{
    struct ast* a;
    tData td;
}
%token EOL T_ADD T_KICK T_TAKE
%token <td> NUM_INT ATOM NUM_DOUBLE  T_BOOL
%token ID

%left T_IN T_CONTAINS
%left T_UNION
%left T_INTER
%left T_DIFF
%left T_CONCAT
%left T_FROM T_TO
%left '+' '-'
%left '*' '/'


%type <a> exp stm 
%type <a> lit_struct list_exp
%start tree


%%
tree: 
| tree stm EOL { printf("=> "); mostrarData(eval($2));printf("\n\n\n"); }
;
stm: exp    {}
;
exp: NUM_INT    { $$ = newast(INT, NULL, NULL, $1);     }   
| ATOM          { $$ = newast(STR, NULL, NULL, $1);     }
| NUM_DOUBLE    { $$ = newast(DOUBLE, NULL, NULL, $1);  }
| T_BOOL        { $$ = newast(BOOL, NULL, NULL, $1);    }
| lit_struct    { $$ = $1;}    

| exp '+' exp   { $$ = newast('+',$1,$3,NULL); }
| exp '-' exp   { $$ = newast('-',$1,$3,NULL); }
| exp '*' exp   { $$ = newast('*',$1,$3,NULL); }
| exp '/' exp   { $$ = newast('/',$1,$3,NULL); }

| T_ADD exp T_TO exp    { $$ = newast(ADD, $2, $4, NULL);   }
| T_KICK exp T_FROM exp { $$ = newast(KICK, $2 , $4, NULL); }
| T_TAKE exp T_FROM exp { $$ = newast(TAKE, $2 , $4, NULL); }
| exp T_CONCAT exp      { $$ = newast(CONCAT,$1,$3, NULL);  }

| exp T_IN exp       { $$ = newast(IN, $1, $3, NULL);       }
| exp T_UNION exp    { $$ = newast(UNION, $1, $3, NULL);    }
| exp T_INTER exp    { $$ = newast(INTER, $1, $3, NULL);    }
| exp T_DIFF exp     { $$ = newast(DIFF, $1, $3, NULL);     }
| exp T_CONTAINS exp { $$ = newast(CONTAINS, $1, $3, NULL); }


lit_struct: '[' ']' { $$ = newast(LIST, NULL, NULL, createData(LIST));  }
| '[' list_exp ']'  { $$ = newast(LIST, $2, NULL, createData(LIST));    }
| '{' '}'           { $$ = newast(SET, NULL, NULL, createData(SET));    }
| '{' list_exp'}'   { $$ = newast(SET, $2, NULL, createData(SET));      }

list_exp: exp       { $$ = $1; }
| exp ',' list_exp  { $$ = newast(LIST_OF_AST, $1, $3, NULL); }

%%
int main(void){
    yyparse();
    //if(yyin != stdin)  fclose(yyin);
    return 0;
}
