class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Característica para divisão
        self.threshold = threshold  # Valor de limiar
        self.left = left            # Subárvore esquerda
        self.right = right          # Subárvore direita
        self.value = value          # Valor (para nós folha)
    
    def is_leaf(self):
        return self.value is not None

class InterpretableTree:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y):
        
        # Implementação real usaria algoritmo mais sofisticado
        self.root = DecisionNode(feature=0, threshold=0.5,
                               left=DecisionNode(value=0),
                               right=DecisionNode(value=1))
    
    def predict(self, x):
        
        node = self.root
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def to_rules(self):
        
        rules = []
        self._traverse(self.root, [], rules)
        return rules
    
    def _traverse(self, node, conditions, rules):
        if node.is_leaf():
            rules.append((conditions.copy(), node.value))
            return
        
        left_cond = f"{node.feature} <= {node.threshold}"
        conditions.append(left_cond)
        self._traverse(node.left, conditions, rules)
        conditions.pop()
        
        right_cond = f"{node.feature} > {node.threshold}"
        conditions.append(right_cond)
        self._traverse(node.right, conditions, rules)
        conditions.pop()