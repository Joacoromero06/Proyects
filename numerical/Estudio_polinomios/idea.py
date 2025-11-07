class Coeffients:
    def __init__(self, coeffs: List[Coefficient]):
        self._coeffs = coeffs.copy()
        self._unified = False

    def unify_type(self)->None:
        """ Modifica in-place el tipo de los coeffs al mas complejo """
        if self._unified or not self._coeffs:
            return
        
        dominant_type = self.get_dominant_type()
        self._coeffs = [ c.promote_to(dominant_type) for c in self._coeffs ]
        self._unified = True

    def get_dominant_type(self)->str:
        """ Determina el tipo mas complejo de los coeff en la _coeffs"""
        map_h = {# hash(tipo) -> hierarchy
            'Integer': 1,
            'Rational': 2,
            'Real': 3
        }

        dominant_type = ''
        dominant_level = 0
        for c in self._coeffs:
            tipo = c.get_type()
            level = map_h[ tipo ] 
            if level > dominant_level:
                dominant_level = level
                dominant_type = tipo
        
        return dominant_type
