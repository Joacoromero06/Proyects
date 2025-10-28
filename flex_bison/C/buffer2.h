# include <stdio.h>
# include <stdlib.h>
# include <string.h>

typedef struct yy_buffer2_state
{
    /* 
    El buffer 2 (pequeño buffer): simula un vector de caracteres
    */
    char* buf;
    int buf_size;
    int buf_len;
}YY_ESTADO_BUFFER_2;

typedef YY_ESTADO_BUFFER_2* YY_BUF_2;

YY_BUF_2 crea_buffer_2( int );
void agrega_string( YY_BUF_2, char* );