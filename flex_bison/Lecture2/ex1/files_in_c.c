/*===================================================================*/
/*=                     ANTICUATE MAIN                              =*/
/*===================================================================*/
/*
*   The classic antiquate version to declare main before ANSI C
*   main(argc, argc)
*   int argc;
*   char ** argv;
*   {//body}
*/


/*===================================================================*/
/*=                      HOW IT WORKS                               =*/
/*===================================================================*/
/*
*   Separates the parameter list from type declaration
*   When sb run './program my_file.txt:
*       argc == 2 && argv[0] == "./program" && argv[1] == "my_file.txt"
*/

/*===================================================================*/
/*=                      WHAT F&B DOES?                             =*/
/*===================================================================*/
/*
*   Flex and Bison generates C well-standared structed programs
*   We compile that c-program to an executable
*   We run the executable and give an input namefile
*/

/*===================================================================*/
/*=                         FILE* yyin                              =*/
/*===================================================================*/
/*
*   Is an global file pointer variable, declarate only by flex produced programs
*   By default is set to 'stdin'
*   Is no static, is: FILE* yyin = NULL;
*/

/*===================================================================*/
/*=                 FILE* fopen(char*, char*)                       =*/
/*===================================================================*/
/*
*   Open a file whose name is given.
*   Setting an mode that let write/read/write&read/etc
*/

/*===================================================================*/
/*=                     perror() function                           =*/
/*===================================================================*/
/*
*   Is a C standard library function
*   Prints a system error corresponding to the las error in (errno) to stderr
*   Used to error reportinf, is tied to errno
*/

typestruct NODE_BUF_STACK{
    struct bufs_stack* prev;    /*puntero a nodo previo */
    //YY_BUFFER_STATE bs;         /*buffer de flex */
    char* filename;             /*nombre del archivo */
    //FILE* f;                    /*puntero al archivo */
    int lineno;                 /*numero de linea escaneadas */
} node_bufs_stack;
typedef node_bufs_stack* BUFF_STACK;
typedef struct NODE_BUFF_STACK* BUFF_STACK;
 

