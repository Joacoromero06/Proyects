#include <iostream>
#include <cstdlib>
#include <ctime>
//#include <string>

int random_int();
std:: string random_string();
float random_float();

template <class Type, class Generator>
void fill_list(Type list[], int len, Generator gen)
{
    for(int i = 0; i < len; i++)
        list[i]  = gen();
}

template <class Type>
void print_list(Type list[], int len)
{
    for (int i = 0; i < len; i++)
        std:: cout << list[i] << " ";
    std:: cout << "\n";
}

template <class Type>
void sort(Type list[], int len)
{
    for(int i = 0; i < len-1; i++)
        for(int j = i+1; j < len; j++)
            if(list[j] < list[i])
            {
                Type temp = list[i];
                list[i] = list[j];
                list[j] = temp;
            }
    
}


int main()
{
    const int len = 100;

    float floats[len];
    int ints[len];
    std:: string strings[len];

    fill_list(strings, len, random_string);
    fill_list(floats, len, random_float);
    fill_list(ints, len, random_int);
    
    sort(strings, len);
    sort(ints, len);
    sort(floats, len);

    print_list(strings, len);
    print_list(ints, len);
    print_list(floats, len);

    return 0;
}

float random_float()
{
    return static_cast<float>( rand() ) / RAND_MAX;
}

int random_int()
{
    return rand() % 100;
}

std:: string random_string()
{
    int s_len = rand() % 5 + 3;
    std:: string s;
    for(int i = 0; i < s_len; i++)
        s.push_back('a' + rand() % 26);
    return s;
}