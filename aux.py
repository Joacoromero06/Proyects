

#!/usr/bin/env python3

def demostrar_error_punto_flotante():
    print("=== DEMOSTRACIÓN DEL ERROR DE PUNTO FLOTANTE ===\n")
    
    # Caso 1: División que produce decimal infinito
    print("1. División con decimal infinito:")
    archivo = 10.0
    velocidad = 3.0
    tiempo = archivo / velocidad
    resultado = archivo - velocidad * tiempo
    
    print(f"   Archivo: {archivo} MB")
    print(f"   Velocidad: {velocidad} MB/s")
    print(f"   Tiempo calculado: {tiempo}")
    print(f"   Archivo - (velocidad * tiempo) = {resultado}")
    print(f"   ¿Es exactamente cero? {resultado == 0.0}")
    print(f"   Valor real: {resultado:.20e}\n")
    
    # Caso 2: Acumulación de errores
    print("2. Acumulación de errores en iteraciones:")
    valor = 0.1
    suma = 0.0
    
    for i in range(10):
        suma += valor
    
    esperado = 1.0
    print(f"   0.1 sumado 10 veces: {suma}")
    print(f"   Valor esperado: {esperado}")
    print(f"   Diferencia: {suma - esperado:.20e}")
    print(f"   ¿Es exactamente 1.0? {suma == 1.0}\n")
    
    # Caso 3: Simulación del problema en nuestro algoritmo
    print("3. Simulación del problema del algoritmo:")
    archivos = [15.7, 23.3, 41.1]  # Tamaños problemáticos
    velocidad_total = 7.3
    
    for i, tamaño in enumerate(archivos):
        velocidad_individual = velocidad_total / len(archivos)
        tiempo_para_completar = tamaño / velocidad_individual
        
        # Simular actualización
        nuevo_tamaño = tamaño - velocidad_individual * tiempo_para_completar
        
        print(f"   Archivo {i+1}: {tamaño} MB")
        print(f"   Velocidad individual: {velocidad_individual:.10f} MB/s")
        print(f"   Tiempo para completar: {tiempo_para_completar:.10f} s")
        print(f"   Tamaño después de restar: {nuevo_tamaño:.20e}")
        print(f"   ¿Sería removido con >0? {nuevo_tamaño > 0}")
        print(f"   ¿Sería removido con >1e-9? {nuevo_tamaño > 1e-9}")
        print()

def comparar_tolerancias():
    print("=== COMPARACIÓN DE DIFERENTES TOLERANCIAS ===\n")
    
    # Valores típicos de error de punto flotante
    errores_tipicos = [
        -1.7763568394002505e-15,  # Error típico en double precision
        2.220446049250313e-16,    # Epsilon de máquina para double
        -1.1102230246251565e-16,  # Otro error común
        5.551115123125783e-17     # Error más pequeño
    ]
    
    tolerancias = [0, 1e-12, 1e-9, 1e-6]
    
    print("Errores típicos vs diferentes tolerancias:")
    print(f"{'Error':<25} {'> 0':<8} {'> 1e-12':<10} {'> 1e-9':<10} {'> 1e-6':<10}")
    print("-" * 65)
    
    for error in errores_tipicos:
        resultados = []
        for tol in tolerancias:
            if tol == 0:
                resultados.append(str(error > 0))
            else:
                resultados.append(str(error > tol))
        
        print(f"{error:<25.2e} {resultados[0]:<8} {resultados[1]:<10} {resultados[2]:<10} {resultados[3]:<10}")

if __name__ == "__main__":
    demostrar_error_punto_flotante()
    print("\n" + "="*60 + "\n")
    comparar_tolerancias()
    
    print("\n=== CONCLUSIÓN ===")
    print("• Los errores de punto flotante son inevitables")
    print("• Usar '> 0' puede dejar archivos 'fantasma' con tamaño negativo infinitesimal")
    print("• Una tolerancia como 1e-9 elimina estos errores sin afectar valores reales")
    print("• 1e-9 es mucho mayor que los errores típicos (~1e-15) pero mucho menor")
    print("  que cualquier tamaño de archivo significativo en el problema")