fn sumar(a: i32, b: i32) -> i32{
    a + b
}
fn multiplicar(a: f64, b: f64) -> f64{
    a * b
}
fn main(){
    let rtdo = sumar(2, 3);
    println!("La suma es {}", rtdo);
    
    
    let rtdo = multiplicar(2.0, 3.0);
    println!("La multiplicacion es {}", rtdo);
}
