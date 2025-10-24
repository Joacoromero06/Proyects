# include "buffer2.h"

YY_BUF_2 crea_buffer_2( int tam )
{
    /* 
    Descripcion:
        Asigno memoria para 'b' el puntero al registro estado del buffer
        Asigno al estado del buffer el tamaño, y asigno 'tam' bytes 
        para almacenar los caracteres en 'yy_ch_buf'
    */
    YY_BUF_2 b = malloc(sizeof( YY_ESTADO_BUFFER_2 ));
    if (!b){perror("malloc: "); exit(1);}

    b->buf_size = tam;
    b->buf = (malloc( tam ));
    if (!b->buf){perror("malloc: "); exit(2);}
    b->buf[0] = '\0';
    b->buf_len = 1;
    return b;
}

void agrega_string( YY_BUF_2 b, char* str)
{
    /*
    descripcion:
        Agrega al final del buffer el string str
        Maneja el overflow del espacio del buffer con salida de error
        Maneja el ultimo caracter nulo '\0'
    */
    int tam_str = strlen(str);
    if (b->buf_len + tam_str <= b->buf_size) 
    {
        memcpy(b->buf + b->buf_len, str, tam_str);
        b->buf_len += tam_str;
        b->buf[b->buf_len] = '\0';
    }
}