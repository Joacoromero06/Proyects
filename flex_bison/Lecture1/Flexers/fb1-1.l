/* Declaracion de variables */
%{
int c_words, c_lines, c_chars;
%}

/* Definicion de patrones y acciones */
%%

[^ \t\n]   { c_words++; c_chars += strlen(yytext); }
\n          { c_lines++; }
.           { c_chars += strlen(yytext); }

%%

int main(){
    yylex();
    printf("Cantidad de lineas%10d\n", c_lines);
    printf("Cantidad de palabras%10d\n", c_words);
    printf("Cantidad de caracteres%10d\n", c_chars);
    return 0;
}