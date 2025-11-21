# include "eval.h"
# include "defs.h"

tData eval(struct ast* a){
    tData d = NULL;
    switch (a->nodetype)
    {
    case INT: d = evaldata( (struct data*) a ); break;
    case LIST: d = evallist( (struct data*) a ); break;
    case ADD: case KICK: d = evalopslist(a); 
    /*case IF: case WHILE: case FORALL: case FORANY: d = evalflow( (struct flow*) a );break;*/   
    default: printf("nodetype desconocido: %d", a->nodetype); break;
    }
    return d;
}

tData evaldata(struct data* a){
    printf("Estoy en evaldata: ");mostrarData(a->d);printf("\n");
    return a->d;
}

tData evallist(struct data* a){
    if (!a->l) return a->d; //empty list_of_ast => empty list
    
    printf("Estoy en evallist\n");tData aux = NULL;

    struct ast* nav     = a->l;
    tData* list_to_load = &(a->d);
    while( nav->nodetype == LIST_OF_AST ){
        aux = eval(nav->l);
        printf("En evallist el elemento a agregar es: "); mostrarData(aux); printf("\n");
        agregarData( list_to_load, aux );
        nav = nav->r;
    }
    agregarData( list_to_load, eval(nav) ); // add the last elemnt in list_of_ast to "lista_a_cargar"

    return a->d;
}

tData evalopslist(struct ast* a){
    tData nuevo;
    struct ast* l=a->l;
    struct ast* r=a->r;
    switch (a->nodetype)
    {
    case ADD: if(r->nodetype == LIST){//cuando es LIST es struct data casteado a ast  3 add [1, 2]
        nuevo=createList();
        nuevo=copiarData(eval(r));
        //nuevo=copiarData(((struct data*) r)->d);
        agregarData(&nuevo, eval(l));
        //agregarData(&nuevo,((struct data*)l)->d);
    }
    break;
    case KICK: if(r->nodetype == LIST){ //cuando es LIST es struct data casteado a ast  3 add [1, 2]
        nuevo = eval(r);                // quiero probar que pasa
        elimniar_pos(&nuevo, eval(l));
    }
        break;
    default: printf("\nExplosion en 3, 2, 1...");
        break;
    }
    return nuevo;
}

tData evalopsint(struct ast* a){
    //oya
}


/*
tData evalflow(struct flow* f){
    tData value = NULL;
    printf("Estoy en un control uy uy uy\n");

    switch(f->nodetype){
        int booleano = 0;

        case IF:
        booleano = get_Bool_Data( eval(f->cond));

        if(booleano != 0){
           value = eval(f->l);
        }
        else{
            if(f->r){
            value = eval(f->r);
            }
        }
        break;

        case WHILE:
        booleano = get_Bool_Data( eval(f->cond) );

        while(booleano != 0){
            value = eval(f->l);
            booleano = get_Bool_Data( eval(f->cond) );
        }

        break;

        case FORALL:
        tData collection = copiarData( eval (f->l) );
        tData iterator = collection;
        while (iterator){
            tData elemento = get_dato(iterator);
            //habria que colocar el id a la tabla de simbolos no entiendo eso todavia jajajajjajaj nt lookup(elem, f->s) o algo asi
            if(f->cond){
                booleano = get_Bool_Data( eval(f->cond) );
            }
            else {
                booleano = 1;
            }

            if(booleano != 0){
                value = eval (f->r);
            }
            iterator = get_next(iterator);
        }
        free(collection);

        break;

        case FORANY:
        tData collection = copiarData( eval (f->l) );
        tData iterator = collection;
        int found = 0;

        while (iterator && found == 0){
            tData elemento = get_dato(iterator);
            //habria que colocar el id a la tabla de simbolos no entiendo eso todavia jajajajjajaj nt lookup(elem, f->s) o algo asi
            if(f->cond){
                booleano = get_Bool_Data( eval(f->cond) );
            }
            else {
                booleano = 1;
            }

            if(booleano != 0){
                found = 1;
                value = eval (f->r);
            }
            iterator = get_next(iterator);
        }
        free(collection);

        break;
        default: printf("pq carajeanos saldria este cartel no");
    }
    return value;

}*/
/*

void treefree(struct ast* a)

}*/