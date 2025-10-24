#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct symbol{
    struct symbol* next;
    char* name;
    struct ref* reflist;
};
struct ref {
    struct ref* next;
    int flag;
    int lineno;
    char* filename;
};
#define NHASH 9997
struct symbol* symtab[NHASH];
struct symbol* lookup(char*);
void addref(int, char*, char*, int);
void printrefs();
char* curfilename;
static unsigned int symhash(char* sym);
int main(){
    int i, n, m;
    //lookup de 3 lexemas iguales
    n = 3;
    for (i=0;i<n;i++){
        struct symbol* sp = lookup("hola");
    }
    
    //lookup de lexemas con mismo hashes que "hola"->79234
    printf("======================MAIN======================\n");
    
    printf("iola->%d, hola->%d, jola->%d\n",symhash("hola "),symhash("hola"),symhash("holaK"));
    lookup("holb");
    lookup("imla");
    lookup("jnka");
    struct symbol** spp = &symtab[symhash("hola")%NHASH];
    if (!spp) printf("spp es null");
    else{
        printf("direccion de la cubeta de \"hola\": %p\n", spp);
        printf("%p\n", *spp);
    }
    struct symbol* sp = *spp;
    i=1;
    while (sp){
        printf("elemento:%s\tn°:%d\n", sp->name, i);
        i++;
        sp=sp->next;
    } 
    return 0;
}
static unsigned int symhash(char* sym){
    unsigned int c, hash = 0;
    
    while( (c= *sym) != '\0' ){
        hash = hash * 9 ^ c;
        sym ++;
    }

    return hash;
}
struct symbol* lookup(char* sym){
    struct symbol** spp = &symtab[symhash(sym)%NHASH];
    printf("=============LOOKUP=============\n");
    printf("%p\n",spp);
    struct symbol* sp = *spp;
    printf("%p\n",sp);
    int scount = NHASH - 1;

    while(scount >= 0){
        if(!sp){
            printf("creando cubeta\n");
            *spp = malloc(sizeof(struct symbol));
            printf("direccion de la cubeta de \"%s\": %p\n", sym, spp);
            (*spp)->name = strdup(sym);
            (*spp)->reflist = NULL;
            printf("%p\n",*spp);

            return *spp;
        } 
        if(sp->name && !strcmp(sym, sp->name)){
            printf("ya esta en symtab\n");
            return sp;
        }
        if(sp->name){//ya hay elementos en la cubeta
            while((sp = sp->next) && strcmp(sym, sp->name));//voy hasta el final
            if(sp) return sp;                                //si lo encontre en la lista de la cubeta
                                                             //no esta en la lista de la cubeta lo agrego
            struct symbol* sp_new = malloc(sizeof(struct symbol));
            sp_new->name = strdup(sym);
            sp_new->reflist = NULL;
            sp->next = sp_new;
            return sp_new;
        }
                  
        if(spp == &symtab[NHASH-1]) 
            spp = symtab;
        else
            spp ++;
        scount --;
    }
    fputs("Error fatal: tabla llena\n", stderr);
    abort();
}