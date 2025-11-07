# Proyect_TPT
Modelo Automata Finito

# Estructura del Proyecto
Decidimos usar esta estructura vista en otros proyectos
Una modificacion nuestra es la carpeta bin dentro de src (codigo fuente)
No parece ser correcto conceptualmente, pero es mas comodo ver el codigo binario ejecutable unico del main
Separamos los archivos dependiendo de su funcion en el proyecto

# Gitignore
Es un archivo propio de haber clonado el repositorio de github
Nos facilita evitar la eliminacion de archivos generados por la compilacion de archivos fuente c
C siempre al compilar genera el ejecutable, en otros IDEs mas especializados, menos libres y flexibles empeora

# Resolucion del Problema
El problema se fue resolviendo por partes.
DATA: 
primero entendimos el modelo matematico inicial Arbol, maneja jerarquias,etc. Rapidamente fuimos a pensar como seria el TAD y sus operaciones, esta parte nos dio el profe. Nos quedaba implementarlo. Lista enlazada de Lista enlazada y Funciones y procedimientos. Para los que fuimos desarrollando y refinando 3 veces cada algoritmo hasta llevarlo a C
AF:
en clases entendimos que era un af y nos basamos en el mismo libro para no cometer errores, Pensamos como seria su TAD su modelo ya lo conociamos bien y sus algoritmos tambien. Diseñamos el TAD pensando en como relacionarlo exclusivamente con sus operaciones. Lo implementamos con un registro, una forma de portar y llevar el AF, habia infinitas formas de implementarlo, la dificultad fue menor en sus operaciones porque sus operaciones son combinaciones de operaciones y operandos que ya modelamos. Tuvimos algunas correcciones en una funcion del modelo anterior, nos olvidamos un caso que era precondicion de ella que no podia atender ese dato.

# Algoritmos
La mayoria de algoritmos eran sencillos, y podiamos desarrollarlos nosotros. Aquello referido al modelo automata
Eran mas complejos y abstractos al venir de un modelo matematico nuevo y por eso usamos los pseudocodigos del libro
Teoria de Automatas Lenjuajes y Computacion

# Como compilar 
gcc test.c Apps/af.c Apps/data.c Apps/pila.c Apps/str.c -IAssets -o bin/test.exe

gcc	Invoca el compilador de C (GNU Compiler Collection).
test.c	Tu archivo principal, con main() probablemente.
Apps/af.c ...	Archivos fuente adicionales necesarios que implementan funciones.
-IAssets	Le dice al compilador que busque los headers (.h) en la carpeta Assets/.
-o bin/test.exe	El ejecutable compilado se guarda en bin/test.exe.

# Como ejecutar
El archivo .exe es el ejecutable generado por el compilador gcc.
Es un archivo binario que tu sistema operativo (Windows en este caso) puede ejecutar directamente.
Se crea con el flag -o.
El ./ es porque el ejecutable está en la carpeta actual (.).

./bin/test.exe. Si estás en src/ y el .exe está en src/bin/test.exe

#({q0,q1},{0,1},{[q0,0,q0],[q0,1,q1],[q1,0,q1],[q1,1,q0]},q0,{q1})
