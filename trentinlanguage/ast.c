#include "ast.h"
#include "data.h"
#include "defs.h"
int b = 0;
int get_nodetype(struct ast *a)
{
    if (!a)
    {
        yyerror("Error en get_nodettype() puntero nulo");
        return 0;
    }
    return a->nodetype;
}

struct ast *newast(int nodoValue, struct ast *left, struct ast *right, tData data)
{
    struct ast *nuevo = malloc(sizeof(struct ast));
    if (!nuevo)
    {
        yyerror("sin memoria en newast");
        return NULL;
    }
    nuevo->nodetype = nodoValue;
    nuevo->l = left;
    nuevo->r = right;
    nuevo->d = data;
    return nuevo;
}

tData eval_list(struct ast *a)
{
    // a = newast(LIST, list_exp, NULL, createData(LIST));
    // listt_exp = newast(LIST_OF_AST, $1, $3, NULL);

    if (!a->l)
        return a->d; // lista vacia se va directo
    tData *list_to_load = &(a->d);
    struct ast *nav = a->l;
    
    while (get_nodetype(nav) == LIST_OF_AST)
    {
        agregarData(list_to_load, eval(nav->l));
        nav = nav->r;
    }
    agregarData(list_to_load, eval(nav));
    return (*list_to_load);
}

tData eval_set(struct ast *a)
{
    // a = newast(LIST, list_exp, NULL, createData(LIST));
    // listt_exp = newast(LIST_OF_AST, $1, $3, NULL);

    if (!a->l)
    {
        return a->d; // lista vacia se va directo
    }
    tData *set_to_load = &(a->d);
    struct ast *nav = a->l;
    while (get_nodetype(nav) == LIST_OF_AST)
    {
        agregarData(set_to_load, eval(nav->l));
        nav = nav->r;
    }
    agregarData(set_to_load, eval(nav));
    return (*set_to_load);
}

tData evalOpList(struct ast *a)
{
    // $$ = newast(KICK, $2 , $4, NULL);
    struct ast *left = a->l;
    struct ast *right = a->r;
    tData aux_result = NULL;
    tData list_result = NULL;
    tData nuevo = NULL;
    aux_result = eval(left);
    list_result = eval(right);
    switch (get_nodetype(a))
    {
    case KICK:
        if (get_tipo(aux_result) == INT && get_tipo(list_result) == LIST)
        {
            nuevo = (list_result);
            int posicion = get_value(eval(left));
            eliminar_pos(&nuevo, posicion);
        }
        else
            printf("no entre al if\n");
        break;
    case ADD:
        if (aux_result != NULL && get_tipo(list_result) == LIST)
        {
            nuevo = list_result;
            agregarData(&nuevo, aux_result);
        }
        else
            printf("Error papu\n");
        break;
    case TAKE:
        if (get_tipo(aux_result) == INT && get_tipo(list_result) == LIST)
        {
            int posicion = get_value(aux_result);
            nuevo = elemento_pos(list_result, posicion);
        }
        else
            printf("error papu\n");
        break;
    case CONCAT:
        if(get_tipo(aux_result)==LIST && get_tipo(list_result) == LIST){
            nuevo=concat_list(aux_result,list_result);
        }
        else printf("Solo se concatenan listas.\n");
        break;
    default:
        printf("\nError papu 2.");
        break;
    }
    return nuevo;
}
tData evalOpSet(struct ast *a)
{
    struct ast *left = a->l;
    struct ast *right = a->r;
    tData conj_1 = eval(left);
    tData conj_2 = eval(right);
    tData nuevo = NULL; 
    if (get_tipo(conj_1) != SET || get_tipo(conj_2) != SET)
    {
        printf("Operacion valida solo para conjuntos\n");
        return NULL;
    }
    switch (get_nodetype(a))
    {
    case UNION:
        nuevo = Union(conj_1, conj_2); 
        break;
    case DIFF:
        nuevo = Diferencia(conj_1,conj_2);
        break;
    case INTER :
        nuevo=Interseccion(conj_1,conj_2);
        break;
     default: printf("Operacion no encontrada papu.\n");
        break;
    }
    return nuevo;
}
tData eval(struct ast *a)
{
    // printf(" MUERE AQUI 0 ");
    struct ast *right = a->r;
    struct ast *left = a->l;
    tData nuevo;
    // printf(" MUERE AQUI 1 ");

    switch (get_nodetype(a))
    {
    case INT:
    case STR:
    case DOUBLE: /*newast(INT, NULL, NULL, $1)*/
    case BOOL:
        if (!a->d)
        {
            printf("ERROR: Literal sin dato\n");
            exit(1);
        }
        nuevo = a->d; // copiar? free?

        break;
    case LIST:
        nuevo = eval_list(a);
        break;
    case SET:
        nuevo = eval_set(a);
        break;
    case '+':
    case '-':
    case '*':
    case '/':
    {
        tData eval_r, eval_l; // printf(" MUERE AQUI 3 ");
        eval_r = eval(right);
        eval_l = eval(left);
        if (get_tipo(eval_r) != BOOL && get_tipo(eval_r) != STR && get_tipo(eval_r) != LIST && get_tipo(eval_r) != SET && get_tipo(eval_l) != BOOL && get_tipo(eval_l) != STR && get_tipo(eval_l) != SET && get_tipo(eval_l) != LIST)
        {
            switch (get_nodetype(a))
            {
            case '+':
                nuevo = sumaData(eval_l, eval_r);
                break;
            case '-':
                nuevo = restaData(eval_l, eval_r);
                break;
            case '*':
                nuevo = prodData(eval_l, eval_r);
                break;
            case '/':
                nuevo = cocData(eval_l, eval_r);
                break;
            default:
                break;
            }
        }
        else
            printf("Error semantico: Solo operaciones entre numeros.");
        break;
    }
    case KICK:
    case ADD:
    case CONCAT:
    case TAKE:
        nuevo = evalOpList(a);
        break;
    case UNION:
    case INTER:
    case DIFF:
        nuevo = evalOpSet(a);
        break;
    /*case IN :
    tData aux1=eval(left); 
    tData aux2=eval(right);
     if(aux1 !=NULL && (get_tipo(aux2)==SET || get_tipo(aux2)==LIST)){
        int rdo = pertenece_completing(aux1,aux2);
        if(rdo=1) nuevo=createBool("true");
        else nuevo=createBool("false");
     }
     else printf("1 no es un elemento permitido o 2 no es una lista.\n"); 
    break*/;
    default:
        printf("Error papu.");
        break;
    }
    return nuevo;
}

int yyerror(char *s)
{
    fprintf(stderr, "Error de sintaxis: %s", s);
}