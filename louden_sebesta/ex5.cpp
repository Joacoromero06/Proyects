#include <iostream>

class Complex{
private:
    float re;
    float im;
public:
    Complex() : re(0), im(0) {}
    Complex(float r, float i) : re(r), im(i) {}

    float real() const {return re;}
    float imag() const {return im;}
  
    Complex operator+ (const Complex& second)
    {
        return Complex( re + second.re , im + second.im );
    }
    

};
std::ostream& operator<< (std::ostream& os, const Complex& c)
{
    os << "(" << c.real() << ", " << c.imag() << ")";
    return os;
}
int main(){
    Complex a = Complex(123. , .123);
    Complex b = Complex(32.2 , 2.10);

    std:: cout << a << " ";
    std:: cout << b << "\n";
    
    return 0;
}