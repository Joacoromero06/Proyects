#include <iostream>
using namespace std;

int sumar(int a, int b)
{
    return a + b;
}
int i;
int main(){
    int suma = sumar(10 , 2);
    int i = 1000;
    cout << "la suma es: " << suma << endl;
    
    for(int i = 0; i < 10; i++){
        cout << "the value of i: " << i << endl;
    }
    ::i = 0;
    cout << "local i: " << i << endl;
    cout << "global i: " << ::i << endl;
    return 0;
}
