class Ontology:
    def __init__(self):
        self.classes = {}
        self.properties = {}
    
    def add_class(self, class_name, properties=None):
        
        if properties is None:
            properties = []
        self.classes[class_name] = properties
    
    def add_property(self, property_name, domain, range):
        
        self.properties[property_name] = {
            'domain': domain,
            'range': range
        }
    
    def get_class_properties(self, class_name):
        
        return self.classes.get(class_name, [])
    
    def load_from_owl(self, file_path):
    
        # Implementação real usaria owlready2
        print(f"Carregando ontologia de {file_path}")
        # Exemplo simplificado
        self.add_class("Animal", ["hasLegs", "canFly"])
        self.add_class("Mammal", ["hasFur"])
        self.add_property("isTypeOf", "Mammal", "Animal")