def suma(self, y: 'MantisaDecimal_10') -> 'MantisaDecimal_10':
        t_self = self.dig_manejados_ent + self.dig_manejados_dec
        t_y = y.dig_manejados_ent + y.dig_manejados_dec
        n = max( t_self, t_y )
        dif = self.dig_manejados_dec - y.dig_manejados_dec
        if dif != 0:
            if dif > 0:
                y.m_decimal.extend([0]*dif)
                y.actualiza()
                #print(f'\nDEBUG1{y}')
            else:
                self.m_inv.extend([0]*abs(dif))
                self.actualiza(False, True, True)
                #print(f'\nDEBUG2{self.m_inv}')

        suma = self.copia()
        acarreo = 0
        for i in range(n):
            xi_mas_yi = self.m_inv[i] + y.m_inv[i]
            en_i = xi_mas_yi + acarreo
            suma.m_inv[i] = en_i % 10 
            acarreo = 1 if en_i >= 10 else 0
            #print(f'\nITERACION = {i}\n self.m_inv[i]: {self.m_inv[i]} + y.m_inv[i]: {y.m_inv[i]}+ acarreo: {acarreo}= en_i: {en_i}')
        if acarreo != 0:
            pass
            #print('EN SUMA, LAS PARTE ENTERAS NO ERAN: 0 DIERON: ',acarreo )
            #suma.append( acarreo )

        suma.actualiza(False, True, True)
        return suma