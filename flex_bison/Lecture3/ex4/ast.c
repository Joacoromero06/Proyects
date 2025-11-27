#include "ast.h"
#include "tree.tab.h"
#include <math.h>
struct symbol symtab[NHASH];

static unsigned int symhash(char* sym){
    unsigned int hash = 0;
    int c;
    while(c = *sym++) hash = hash * 9 ^ c;
    return hash;
}

struct symbol* lookup(char* sym){
    struct symbol* sp = &symtab[symhash(sym)%NHASH];
    int scount = NHASH;
    while (--scount >= 0){
        if(sp->name && !strcmp(sp->name, sym)) return sp;
        if(!sp->name){
            sp->name = strdup(sym);
            sp->value = 0;
            sp->fn = NULL;
            sp->sl = NULL;
            return sp;
        }
        if(++sp >= symtab + NHASH) sp = symtab;
    }
    yyerror("\noverflow tabla de symbolos"); abort();    
}

struct ast* newast(int nodetype, struct ast* l, struct ast* r){
    struct ast* a = malloc(sizeof(struct ast));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = nodetype; a->l = l; a->r = r; 

    return (struct ast*) a;
}

struct ast* newcmp(int nodetype, struct ast* l, struct ast* r){
    struct ast* a = malloc(sizeof(struct ast));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = nodetype + '0'; a->l = l; a->r = r;
    return (struct ast*) a;
}

struct ast* newnum(double d){
    struct num* a = malloc(sizeof(struct num));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = 'K'; a->number = d;
    return (struct ast*) a;
}

struct ast* newref(struct symbol* s){
    struct ref* a = malloc(sizeof(struct ref));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = 'R'; a->s = s;
    return (struct ast*) a;
}

struct ast* newasgn(struct symbol* s, struct ast* v){
    struct asgn* a = malloc(sizeof(struct asgn));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = '='; a->s = s; a->v = v;
    return (struct ast*) a;
}

struct ast* newfncall(int fntype, struct ast* params){
    struct fncall* a = malloc(sizeof(struct fncall));
    if(!a){printf("\nSin memoria"); exit(0);}
    
    a->nodetype = 'F'; a->l = params; a->fntype = fntype;
    return (struct ast*) a;
}

struct ast* newufncall(struct symbol* s, struct ast* params){
    struct ufncall* a = malloc(sizeof(struct ufncall));
    if(!a){printf("\nSin memoria"); exit(0);}
    
    a->nodetype = 'U'; a->l = params; a->s = s;
    return (struct ast*) a;
}

struct ast* newflow(int nodetype, struct ast* cond, struct ast* tblock, struct ast* fblock){
    struct flow* a = malloc(sizeof(struct flow));
    if(!a){printf("\nSin memoria"); exit(0);}

    a->nodetype = nodetype; a-> cond = cond; a->tblock = tblock; a->fblock = fblock;
    return (struct ast*) a;
}

void freetree(struct ast* a){
    switch (a->nodetype){
    case '+': case'-': case'*': case'/': case 'L':
    case 1: case 2: case 3: case 4: case 5: case 6: 
        freetree(a->r);
    case 'U': case 'F': case 'M': freetree(a->l);
    case 'K': case 'R': break;
    case '=': free( ((struct asgn*)a)->v ); break; //freetree
    case 'I': case 'W': 
        free( ((struct flow*)a)->cond ); //freetree
        if( ((struct flow*)a)->tblock ) freetree( ((struct flow*)a)->tblock );
        if( ((struct flow*)a)->fblock ) freetree( ((struct flow*)a)->fblock );
        break;
    default:
        printf("\nError interno '%c' '%d'", a->nodetype, a->nodetype);
        break;
    }
    free(a);
} 

struct symlist* newsymlist(struct symbol* s, struct symlist* next){
    struct symlist* sl = malloc(sizeof(struct symlist));
    if (!sl){printf("\nSin memoria"); exit(0);}

    sl->s = s; sl->next = next;
    return sl;
}
void freesymlist(struct symlist* sl){
    struct symlist* next;
    while (sl){
        next = sl->next;
        free(sl);
        sl = next;
    }   
}

static double callbuiltinfunction(struct fncall* f){
    enum bifs type = f->fntype;
    double v = eval(f->l);

    switch(type){
        case F_sqrt: return sqrt(v);
        case F_exp: return exp(v);
        case F_log: return log(v);
        case F_print: return printf("%f",v);
        default: printf("Funcion desconocida"); return 0.0;
    }    
}
static double usercall(struct ufncall* f){
    struct symbol* s = f->s;
    struct symlist* sl = NULL;
    struct ast* args = f->l;

    double* oldvals, * newvals, v;
    int nargs, i;

    if(!s->fn){printf("call to funcion indefinida %s", s->name);return 0.0;}

    //computar nargs
    sl = s->sl;
    for(nargs = 0; sl; sl = sl->next)
        nargs++;
    
    //crear vectores
    oldvals = (double*)malloc(sizeof(double)*nargs);
    newvals = (double*)malloc(sizeof(double)*nargs);
    if(!oldvals | !newvals) { printf("Sin memoria"); return 0.0;}
    
    //computo newvals
    for(i=0; i<nargs; i++){
        if(!args){printf("Pocos argumentos para %s", s->name); free(oldvals); free(newvals); return 0.0;}
        if(args->nodetype == 'L'){
            newvals[i] = eval(args->l);
            args = args->r;
        }
        else newvals[i] = eval(args);
    }

    //Asgino los params actuales
    //Guardo los valores de los args formales
    sl = s->sl;
    for(i=0; i<nargs; i++){
        oldvals[i] = sl->s->value;
        sl->s->value = newvals[i];
        sl = sl->next;
    }
    free(newvals);

    //evaluo el body, nuevos params actuales
    v = eval(s->fn);

    //valores originales de los args formales
    sl = s->sl;
    for(i=0; i<nargs; i++){
        sl->s->value = oldvals[i];
        sl = sl->next;
    }
    free(oldvals);
    return v;
}

double eval(struct ast* a){
    double v;
    if(!a){printf("Error interno, eval ast null"); return 0.0;}

    switch(a->nodetype){
        case '+': v = eval(a->l) + eval(a->r); break;
        case '-': v = eval(a->l) - eval(a->r); break;
        case '*': v = eval(a->l) * eval(a->r); break;
        case '/': v = eval(a->l) / eval(a->r); break;

        case 'M': v = -eval(a->l); break;
        case '|': v = fabs(eval(a->l)); break;

        case 1: v = (eval(a->l) < eval(a->r))? 1 : 0; break;
        case 2: v = (eval(a->l) > eval(a->r))? 1 : 0; break;
        case 3: v = (eval(a->l) == eval(a->r))? 1 : 0; break;
        case 4: v = (eval(a->l) != eval(a->r))? 1 : 0; break;
        case 5: v = (eval(a->l) <= eval(a->r))? 1 : 0; break;
        case 6: v = (eval(a->l) >= eval(a->r))? 1 : 0; break;

        case 'K': v = ((struct num*)a)->number; break;
        case 'R': v = ((struct ref*)a)->s->value; printf("variable, valor:%f\n",v);break;
        
        case '=': v =  eval(((struct asgn*)a)->v); ((struct asgn*)a)->s->value = v; printf("se asigno a la variable: %f\n", v); break;

        case 'I':
            if(eval(((struct flow*)a)->cond)){
                if(((struct flow*)a)->tblock) v = eval(((struct flow*)a)->tblock);
                else v = 0.0;
            }
            else{
                if(((struct flow*)a)->fblock) v = eval(((struct flow*)a)->fblock);
                else v = 0.0;
            }    
        break;
        case 'W':
            v = 0.0; //default
            while (eval(((struct flow*)a)->cond))
                if (((struct flow*)a)->tblock) v = eval(((struct flow*)a)->tblock);
        break;
        case 'L': eval(a->l); v = eval(a->r); break;
        case 'F': v = callbuiltinfunction((struct fncall*)a); break;
        case 'U': v = usercall((struct ufncall*) a); break;
        default: printf("\nError interno, nodo desconocido '%c' '%d'", a->nodetype,a->nodetype );
            
    }
    return v;
}

void dodef(struct symbol* fn, struct symlist* sl, struct ast* a){
    if(fn->sl) freesymlist(fn->sl);
    if(fn->fn) freetree(fn->fn);// the ast of the body function
    fn->sl = sl;
    fn->fn = a;
}   


int yyerror(char* s, ...){
    va_list ap; /*va_list: tipo definido en stdarg, puntero a lista de argumentos*/
    va_start(ap, s); /*Inicializa el puntero al siguiente argumento despues de s*/
    
    fprintf(stderr, "(error %d)", yylineno);
    vfprintf(stderr, s, ap);
    /*
    en el stream stderr, se agrega la cadena de formato s, los valores de los
    formatos estan en ap. 
    */
    fprintf(stderr, "\n");
    va_end(ap);/*free de va_list ap*/
}
int main(){
    printf("Calculadora\n");
    yyparse();
    return 0;
}
