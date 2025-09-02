#include <stdio.h>
#include <stdlib.h>

#define yy_BUF_SIZE 8192 //8kb
#define yy_END_BUFFER  '\0'

typedef struct yy_buffer_state{
    FILE* yy_input_file;    //puntero al archivo de entrada
    char* yy_ch_buf;        //puntero al bloque de memoria 'buffer'
    int yy_ch_buf_size;     //tamaño del buffer
    
    // nro de char validos en el buffer
    int yy_n_chars;         

    char* yy_buf_end;       //fin del token
    char* yy_buf_start;     //inicio del token

    char* yy_buf_pos;       //posicion actual de lectura
    int yy_eof_reached;     //flag de fin de archivo
}YY_BUFFER_STATE;

YY_BUFFER_STATE* yy_create_buffer(FILE *file, int tam){
    YY_BUFFER_STATE* buffer = malloc( sizeof(YY_BUFFER_STATE) );
    if (!buffer) { perror("malloc: "); exit(1);}

    buffer->yy_input_file = file;
    buffer->yy_ch_buf_size = tam;
    buffer->yy_ch_buf = malloc(tam+2);//+2 para centinelas

    if (!buffer->yy_ch_buf) { perror("malloc: "); exit(2);}

    buffer->yy_n_chars = 0;
    buffer->yy_buf_start=buffer->yy_buf_pos=buffer->yy_buf_end=buffer->yy_ch_buf;
    buffer->yy_eof_reached = 0;
    return buffer;
} 
int yy_fill_buffer(YY_BUFFER_STATE *buffer) {
    if ( buffer->yy_eof_reached ){return 0;}

    int num_to_read = buffer->yy_ch_buf_size;

    int char_reads = fread(buffer->yy_ch_buf,1,num_to_read,buffer->yy_input_file);
    buffer->yy_n_chars=char_reads;

    if ( char_reads==0 ){
        buffer->yy_eof_reached=1;
        //sellar el buffer 2 sentinelas
        buffer->yy_ch_buf[0]=buffer->yy_ch_buf[1]=yy_END_BUFFER;
    }
    
    //agregar centinelas
    buffer->yy_ch_buf[char_reads+1]=buffer->yy_ch_buf[char_reads+2]=yy_END_BUFFER;
    
    buffer->yy_buf_pos=buffer->yy_ch_buf;
    return 1;
}
int yy_input(YY_BUFFER_STATE *b){
    if (*(b->yy_ch_buf) = yy_END_BUFFER ){
        if( !yy_fill_buffer(b) ){  return EOF;}
    }
    int c=*(b->yy_ch_buf);
    b->yy_ch_buf++;
    return c;
}


int main(){
    FILE* f= fopen("nombre.txt","r?");
    if(!f){ perror("fopen: "); exit(1); }

    YY_BUFFER_STATE* mi_buffer= yy_create_buffer(f,yy_BUF_SIZE);
    yy_fill_buffer(mi_buffer);

    int c;
    while ( (c=yy_input(mi_buffer)) !=EOF)
    {
        putchar(c);
    }

    fclose(f);
    free(mi_buffer->yy_ch_buf);
    free(mi_buffer);
    return 0;
    
}