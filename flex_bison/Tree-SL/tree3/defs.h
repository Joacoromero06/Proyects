#ifndef DEFS_H
#define DEFS_H

# include <stdio.h>
# include <stdlib.h> 
/*========================================================================*/
/*nodetypes for struct data, distinguishes the different datatypes in Tree*/
/*========================================================================*/
#define STR 1000
#define LIST 1002
#define SET 1003
#define INT 1004
#define DOUBLE 1005
#define BOOL 1006
/*===============================================*/


/*=======================================================================*/
/*nodetypes for ast that represent classic arithmetic expression         */
/*=======================================================================*/
#define SUMA 43
#define RESTA 45
#define PRODUCTO 42
#define COCIENTE 47
/*=======================================================================*/



/*=======================================================================*/
/*nodetypes for ast flows                                                */
/*=======================================================================*/
#define WHILE 1100
#define IF 1101
#define FORALL 1102
#define FORANY 1103
/*=======================================================================*/

/*=======================================================================*/
/*nodetypes for ast that represents list operations                                         */
/*=======================================================================*/
#define ADD 1201
#define KICK 1202
/*=======================================================================*/


/*=======================================================================*/
/*nodetype for ast that represent linked list of ast:                    */
/*      data in left child                                               */
/*      next in right child                                              */
/*      The list finish when nodetype != LIST_OF_AST                     */  
/*=======================================================================*/
#define LIST_OF_AST 1200
/*===============================================*/
int yylex();

#endif

