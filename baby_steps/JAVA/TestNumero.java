import java.util.Scanner;

public class TestNumero{
    public static void main (String args[]) {
        Scanner sc = new Scanner(System.in);
        int x1, x2, fact_x;
        
        System.out.println("Ingrese un numero");
        x1 = sc.nextInt();
        System.out.println("Ingrese un numero");
        x2 = sc.nextInt();

        Numero num1 = new Numero(x1);
        Numero num2 = new Numero(x2);
        

        System.out.println("gcd entre: " + num1.getX() + " y " + num2.getX() );
        System.out.println(num1.GcdIterativo(num2.getX()));
        System.out.println(num1.GcdRecursivo(num2.getX()));
        System.out.println(num1.GcdIterativo2(num2));
        System.out.println(num1.GcdRecursivo2(num2));

        sc.close();
    }
}