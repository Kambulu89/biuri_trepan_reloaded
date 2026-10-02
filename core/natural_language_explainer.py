import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import random
import re

class NaturalLanguageExplainer:
    
    def __init__(self, feature_names: List[str], class_names: List[str]):
        self.feature_names = feature_names
        self.class_names = class_names
        self.tree_model = None
        self.training_data = None
        self.training_labels = None
        
        self.templates = {
            'rule_summary': [
                "Si {conditions}, entonces la predicción es '{prediction}'.",
                "Cuando {conditions}, el modelo clasifica como '{prediction}'.",
                "La regla indica que {conditions} resulta en '{prediction}'.",
                "En caso de {conditions}, la decisión es '{prediction}'.",
                "Para que el modelo clasifique como '{prediction}', es necesario que {conditions}.",
                "El patrón '{prediction}' ocurre cuando {conditions}.",
                "La condición {conditions} conduce a la clasificación '{prediction}'.",
                "Cuando tenemos {conditions}, el resultado es '{prediction}'."
            ],
            'path_explanation': [
                "Para llegar a esta clasificación, el modelo siguió este camino:",
                "La decisión se tomó siguiendo esta secuencia de condiciones:",
                "Este resultado se obtuvo mediante el siguiente proceso:",
                "El modelo llegó a esta conclusión porque:",
                "La clasificación '{prediction}' se determinó mediante este análisis:",
                "Veamos cómo el modelo llegó a esta decisión:",
                "El camino de decisión que llevó a '{prediction}' fue:",
                "Esta es la secuencia de verificaciones que resultó en '{prediction}':"
            ],
            'example_intro': [
                "Por ejemplo:",
                "Un caso típico sería:",
                "Considere este ejemplo:",
                "Imagine un caso donde:",
                "Veamos un ejemplo práctico:",
                "Como ejemplo, tenemos:",
                "Para ilustrar, considere:",
                "Un ejemplo concreto sería:"
            ],
            'confidence_explanation': [
                "El modelo tiene {confidence:.0%} de confianza en esta decisión.",
                "Esta predicción tiene {confidence:.0%} de certeza.",
                "El nivel de confianza para esta clasificación es {confidence:.0%}.",
                "Esta decisión es {confidence:.0%} confiable.",
                "La confianza del modelo en esta predicción es de {confidence:.0%}.",
                "Con {confidence:.0%} de confianza, el modelo clasifica como '{prediction}'.",
                "El grado de certeza de esta clasificación es {confidence:.0%}.",
                "Esta es una decisión {confidence_level} con {confidence:.0%} de confianza."
            ]
        }
        
        self.operator_mapping = {
            'le': 'es menor o igual a',
            'gt': 'es mayor que',
            'lt': 'es menor que',
            'ge': 'es mayor o igual a',
            'eq': 'es igual a',
            'ne': 'es diferente de'
        }
    
    def set_tree_model(self, tree_model, training_data: np.ndarray = None, training_labels: np.ndarray = None):

        self.tree_model = tree_model
        self.training_data = training_data
        self.training_labels = training_labels
    
    def generate_rule_summaries(self, max_rules: int = 10) -> List[Dict[str, Any]]:

        if self.tree_model is None:
            return []
        
        rules = self._extract_all_rules()
        
        rules.sort(key=lambda x: x['samples'], reverse=True)
        
        summaries = []
        for i, rule in enumerate(rules[:max_rules]):
            summary = {
                'rule_id': i + 1,
                'text': self._format_rule_text(rule),
                'prediction': rule['prediction'],
                'confidence': rule['confidence'],
                'samples': rule['samples'],
                'examples': self._generate_examples_for_rule(rule),
                'importance_score': self._calculate_importance_score(rule)
            }
            summaries.append(summary)
        
        return summaries
    
    def generate_path_explanation(self, sample: np.ndarray) -> Dict[str, Any]:
        
        if self.tree_model is None:
            return {}
        
        path = self._get_prediction_path(sample)
        prediction = self.tree_model.predict([sample])[0]
        prediction_name = self.class_names[prediction]
        
        extrapolation_info = self._detect_extrapolation(sample)
        
        explanation = {
            'prediction': prediction_name,
            'confidence': self._calculate_prediction_confidence(sample),
            'path_steps': [],
            'summary': '',
            'examples': [],
            'extrapolation': extrapolation_info
        }
        
        for i, step in enumerate(path):
            step_text = self._format_path_step(step, i + 1)
            explanation['path_steps'].append(step_text)
        
        explanation['summary'] = self._generate_path_summary(path, prediction_name)
        
        explanation['examples'] = self._find_similar_examples(sample, prediction)
        
        return explanation
    
    def _detect_extrapolation(self, sample: np.ndarray) -> Dict[str, Any]:
        
        if self.training_data is None:
            return {'is_extrapolation': False, 'message': 'Datos de entrenamiento no disponibles'}
        
        extrapolation_features = []
        warnings = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= len(sample):
                continue
                
            feature_value = sample[i]
            feature_data = self.training_data[:, i]
            
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)
            
            if feature_value < min_val:
                extrapolation_features.append(feature_name)
                warnings.append(f"{feature_name} ({feature_value:.2f}) está por debajo del mínimo de entrenamiento ({min_val:.2f})")
            elif feature_value > max_val:
                extrapolation_features.append(feature_name)
                warnings.append(f"{feature_name} ({feature_value:.2f}) está por encima del máximo de entrenamiento ({max_val:.2f})")
        
        is_extrapolation = len(extrapolation_features) > 0
        
        if is_extrapolation:
            message = f"ATENCIÓN: Esta muestra extrapola los datos de entrenamiento en {len(extrapolation_features)} características: {', '.join(extrapolation_features)}"
        else:
            message = "OK: Esta muestra está dentro del rango de los datos de entrenamiento"
        
        return {
            'is_extrapolation': is_extrapolation,
            'extrapolation_features': extrapolation_features,
            'warnings': warnings,
            'message': message,
            'confidence_adjustment': 0.1 if is_extrapolation else 0.0
        }
    
    def answer_question(self, question: str, sample: np.ndarray = None) -> Dict[str, Any]:
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['por que', 'por qué', 'why', 'motivo', 'razón', 'causa']):
            return self._answer_why_question(sample)
        elif any(word in question_lower for word in ['como', 'cómo', 'how', 'funciona', 'processo', 'mecanismo']):
            return self._answer_how_question(sample)
        elif any(word in question_lower for word in ['quando', 'cuándo', 'when', 'condição', 'condición', 'critério', 'criterio', 'requisito']):
            return self._answer_when_question()
        elif any(word in question_lower for word in ['onde', 'dónde', 'where', 'local', 'posição', 'posición', 'fonte', 'fuente']):
            return self._answer_where_question()
        elif any(word in question_lower for word in ['o que', 'qué', 'what', 'analisa', 'analiza', 'examina', 'considera']):
            return self._answer_what_question()
        elif any(word in question_lower for word in ['qual', 'cuál', 'which', 'melhor', 'peor', 'diferença', 'diferencia']):
            return self._answer_which_question()
        elif any(word in question_lower for word in ['quantos', 'cuántos', 'how many', 'número', 'numero', 'contagem', 'conteo', 'estatística', 'estadística']):
            return self._answer_how_many_question()
        elif any(word in question_lower for word in ['confiança', 'confianza', 'confidence', 'certeza', 'certeza', 'precisão', 'precisión', 'acurácia', 'exactitud']):
            return self._answer_confidence_question(sample)
        else:
            return self._answer_general_question(question)
    
    def generate_concrete_examples(self, rule_id: int = None, max_examples: int = 5) -> List[Dict[str, Any]]:

        if self.training_data is None or self.training_labels is None:
            return []
        
        examples = []
        
        if rule_id is not None:
            rules = self._extract_all_rules()
            if rule_id <= len(rules):
                rule = rules[rule_id - 1]
                examples = self._generate_examples_for_rule(rule, max_examples)
        else:
            rules = self._extract_all_rules()
            rules.sort(key=lambda x: x['samples'], reverse=True)
            
            for rule in rules[:3]:
                rule_examples = self._generate_examples_for_rule(rule, 2)
                examples.extend(rule_examples)
        
        return examples[:max_examples]
    
    def _extract_all_rules(self) -> List[Dict[str, Any]]:
        
        if self.tree_model is None:
            return []

        if hasattr(self.tree_model, 'root_') and hasattr(self.tree_model, 'export_rules'):
            rules = []
            class_values = np.asarray(getattr(self.tree_model, 'classes_', []))

            def class_name(prediction):
                if class_values.size:
                    indices = np.where(class_values == prediction)[0]
                    if len(indices) and int(indices[0]) < len(self.class_names):
                        return self.class_names[int(indices[0])]
                return str(prediction)

            def walk(node, conditions, native_conditions):
                if node.is_leaf:
                    confidence = float(np.max(node.distribution)) if len(node.distribution) else 0.0
                    rule = {
                        'conditions': list(conditions),
                        'native_conditions': list(native_conditions),
                        'prediction': class_name(node.prediction),
                        'samples': int(len(node.real_y)),
                        'confidence': confidence,
                        'node_id': int(node.node_id),
                        'importance_score': self._calculate_importance_score({
                            'samples': int(len(node.real_y)), 'confidence': confidence
                        }),
                    }
                    rules.append(rule)
                    return
                text = node.test.text(self.feature_names)
                walk(
                    node.false_child, conditions + [f'NÃO {text}'],
                    native_conditions + [(node.test, False)],
                )
                walk(
                    node.true_child, conditions + [text],
                    native_conditions + [(node.test, True)],
                )

            walk(self.tree_model.root_, [], [])
            return rules

        tree = self.tree_model.tree_
        rules = []
        
        def extract_rules_from_node(node_id, conditions):
            if tree.children_left[node_id] == -1:
                prediction = np.argmax(tree.value[node_id][0])
                prediction_name = self.class_names[prediction]
                samples = tree.n_node_samples[node_id]
                confidence = np.max(tree.value[node_id][0]) / np.sum(tree.value[node_id][0])
                
                simplified_conditions = self._simplify_conditions(conditions)
                
                rule = {
                    'conditions': simplified_conditions,
                    'prediction': prediction_name,
                    'samples': samples,
                    'confidence': confidence,
                    'node_id': node_id,
                    'importance_score': self._calculate_importance_score({
                        'samples': samples,
                        'confidence': confidence
                    })
                }
                rules.append(rule)
            else:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = self.feature_names[feature]
                
                left_conditions = conditions + [f"{feature_name} <= {threshold:.2f}"]
                extract_rules_from_node(tree.children_left[node_id], left_conditions)
                
                right_conditions = conditions + [f"{feature_name} > {threshold:.2f}"]
                extract_rules_from_node(tree.children_right[node_id], right_conditions)
        
        extract_rules_from_node(0, [])
        return rules
    
    def _simplify_conditions(self, conditions: List[str]) -> List[str]:

        if not conditions:
            return conditions
        
        feature_conditions = {}
        for condition in conditions:
            parts = condition.split()
            if len(parts) >= 3:
                feature = parts[0]
                operator = parts[1]
                try:
                    threshold = float(parts[2])
                except (ValueError, TypeError):
                    continue
                
                if feature not in feature_conditions:
                    feature_conditions[feature] = []
                
                feature_conditions[feature].append((operator, threshold))
        
        simplified_conditions = []
        for feature, conds in feature_conditions.items():
            simplified = self._simplify_feature_conditions(feature, conds)
            simplified_conditions.extend(simplified)
        
        return simplified_conditions
    
    def _simplify_feature_conditions(self, feature: str, conditions: List[Tuple[str, float]]) -> List[str]:
        
        if len(conditions) == 1:
            operator, threshold = conditions[0]
            return [f"{feature} {operator} {threshold:.2f}"]
        
        le_conditions = [t for op, t in conditions if op == '<=' and isinstance(t, (int, float))]
        gt_conditions = [t for op, t in conditions if op == '>' and isinstance(t, (int, float))]
        
        simplified = []
        
        if le_conditions:
            min_le = min(le_conditions)
            simplified.append(f"{feature} <= {min_le:.2f}")
        
        if gt_conditions:
            max_gt = max(gt_conditions)
            simplified.append(f"{feature} > {max_gt:.2f}")
        
        return simplified
    
    def _format_rule_text(self, rule: Dict[str, Any]) -> str:
        
        conditions = rule['conditions']
        prediction = rule['prediction']
        
        if len(conditions) == 1:
            conditions_text = conditions[0]
        elif len(conditions) == 2:
            conditions_text = f"{conditions[0]} Y {conditions[1]}"
        else:
            conditions_text = f"{', '.join(conditions[:-1])} Y {conditions[-1]}"
        
        template = random.choice(self.templates['rule_summary'])
        return template.format(conditions=conditions_text, prediction=prediction)
    
    def _get_prediction_path(self, sample: np.ndarray) -> List[Dict[str, Any]]:

        if hasattr(self.tree_model, 'root_'):
            path = []
            node = self.tree_model.root_
            row = np.asarray(sample, dtype=float).reshape(1, -1)
            while not node.is_leaf:
                outcome = bool(node.test.evaluate(row)[0])
                path.append({
                    'feature': node.test.text(self.feature_names),
                    'threshold': None,
                    'value': None,
                    'condition': 'sim' if outcome else 'não',
                    'node_id': int(node.node_id),
                    'test_text': node.test.text(self.feature_names),
                })
                node = node.true_child if outcome else node.false_child
            return path

        tree = self.tree_model.tree_
        path = []
        node_id = 0
        
        while tree.children_left[node_id] != -1:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = self.feature_names[feature]
            feature_value = sample[feature]
            
            step = {
                'feature': feature_name,
                'threshold': threshold,
                'value': feature_value,
                'condition': '≤' if feature_value <= threshold else '>',
                'node_id': node_id
            }
            path.append(step)
            
            if feature_value <= threshold:
                node_id = tree.children_left[node_id]
            else:
                node_id = tree.children_right[node_id]
        
        return path
    
    def _format_path_step(self, step: Dict[str, Any], step_number: int) -> str:
        
        feature = step['feature']
        threshold = step.get('threshold')
        value = step.get('value')
        condition = step['condition']
        if step.get('test_text'):
            return f"Passo {step_number}: {step['test_text']} → {condition}"
        if condition == '<=':
            condition_text = f"é menor ou igual a {threshold:.2f}"
        else:
            condition_text = f"é maior que {threshold:.2f}"
        return f"Passo {step_number}: {feature} ({value:.2f}) {condition_text}"
    
    def _generate_path_summary(self, path: List[Dict[str, Any]], prediction: str) -> str:
        
        if not path:
            return f"La muestra fue clasificada directamente como '{prediction}'."
        
        intro = random.choice(self.templates['path_explanation'])
        steps_text = " -> ".join([step['feature'] for step in path])
        
        return f"{intro}\n{steps_text} -> '{prediction}'"
    
    def _calculate_prediction_confidence(self, sample: np.ndarray) -> float:
        
        if self.tree_model is None:
            return 0.5
        
        extrapolation_info = self._detect_extrapolation(sample)
        
        if hasattr(self.tree_model, 'predict_proba'):
            probabilities = self.tree_model.predict_proba([sample])[0]
            base_confidence = np.max(probabilities)
        else:
            base_confidence = 0.8
        
        if extrapolation_info['is_extrapolation']:
            adjusted_confidence = base_confidence - extrapolation_info['confidence_adjustment']
            return max(adjusted_confidence, 0.1)
        
        return base_confidence
    
    def _generate_examples_for_rule(self, rule: Dict[str, Any], max_examples: int = 3) -> List[Dict[str, Any]]:
       
        if self.training_data is None:
            return []
        
        examples = []
        
        matching_samples = self._find_samples_matching_rule(rule)
        
        if not matching_samples:
            return []
        
        selected_indices = random.sample(matching_samples, min(len(matching_samples), max_examples))
        
        for idx in selected_indices:
            sample = self.training_data[idx]
            label = self.training_labels[idx] if self.training_labels is not None else "Desconocido"
            
            example = {
                'sample_id': idx,
                'features': self._format_sample_features(sample),
                'actual_label': label,
                'predicted_label': rule['prediction'],
                'match': label == rule['prediction']
            }
            examples.append(example)
        
        return examples
    
    def _find_samples_matching_rule(self, rule: Dict[str, Any]) -> List[int]:
        
        if self.training_data is None:
            return []
        
        matching_indices = []
        
        for i, sample in enumerate(self.training_data):
            if self._sample_matches_rule(sample, rule):
                matching_indices.append(i)
        
        return matching_indices
    
    def _sample_matches_rule(self, sample: np.ndarray, rule: Dict[str, Any]) -> bool:
        native = rule.get('native_conditions') or []
        if native:
            row = np.asarray(sample, dtype=float).reshape(1, -1)
            for test, outcome in native:
                value = bool(test.evaluate(row)[0])
                if value != bool(outcome):
                    return False
            return True
        for condition in rule['conditions']:
            if not self._evaluate_condition(sample, condition):
                return False
        return True
    
    def _evaluate_condition(self, sample: np.ndarray, condition: str) -> bool:
        
        match = re.match(r'(.+?)\s*(<=|>)\s*([\d.]+)', condition)
        if not match:
            return False
        
        feature_name, operator, threshold = match.groups()
        
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            return False
        
        try:
            feature_idx = self.feature_names.index(feature_name)
            feature_value = sample[feature_idx]
            
            if not isinstance(feature_value, (int, float)):
                try:
                    feature_value = float(feature_value)
                except (ValueError, TypeError):
                    return False
            
            if operator == '<=':
                return feature_value <= threshold
            else:
                return feature_value > threshold
        except (ValueError, IndexError):
            return False
    
    def _format_sample_features(self, sample: np.ndarray) -> Dict[str, float]:
        
        features = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(sample):
                features[feature_name] = float(sample[i])
        return features
    
    def _find_similar_examples(self, sample: np.ndarray, prediction: str, max_examples: int = 3) -> List[Dict[str, Any]]:
        
        if self.training_data is None:
            return []
        
        distances = []
        for i, training_sample in enumerate(self.training_data):
            distance = np.linalg.norm(sample - training_sample)
            distances.append((i, distance))
        
        distances.sort(key=lambda x: x[1])
        
        examples = []
        for i, (idx, distance) in enumerate(distances[:max_examples]):
            training_sample = self.training_data[idx]
            label = self.training_labels[idx] if self.training_labels is not None else "Desconocido"
            
            example = {
                'sample_id': idx,
                'features': self._format_sample_features(training_sample),
                'actual_label': label,
                'similarity': 1.0 / (1.0 + distance)
            }
            examples.append(example)
        
        return examples
    
    def _calculate_importance_score(self, rule: Dict[str, Any]) -> float:
        
        samples_score = min(rule['samples'] / 100.0, 1.0)
        confidence_score = rule['confidence']
        
        return (samples_score + confidence_score) / 2.0
    
    def _answer_why_question(self, sample: np.ndarray) -> Dict[str, Any]:
        
        if sample is None:
            return {'answer': 'Por favor, proporcione una muestra para el análisis.'}
        
        path_explanation = self.generate_path_explanation(sample)
        
        answer = f"El modelo clasificó esta muestra como '{path_explanation['prediction']}' porque:\n\n"
        
        for step in path_explanation['path_steps']:
            answer += f"• {step}\n"
        
        answer += f"\nConfianza: {path_explanation['confidence']:.0%}"
        
        return {
            'answer': answer,
            'path_explanation': path_explanation,
            'question_type': 'why'
        }
    
    def _answer_how_question(self, sample: np.ndarray) -> Dict[str, Any]:
        
        return {
            'answer': 'El modelo usa un árbol de decisión que hace preguntas secuenciales sobre las características de los datos. Cada respuesta conduce a una nueva pregunta hasta llegar a una decisión final.',
            'question_type': 'how'
        }
    
    def _answer_when_question(self) -> Dict[str, Any]:
        
        rules = self._extract_all_rules()
        rules.sort(key=lambda x: x['samples'], reverse=True)
        
        answer = "El modelo hace diferentes clasificaciones cuando:\n\n"
        
        for i, rule in enumerate(rules[:5]):
            answer += f"{i+1}. {self._format_rule_text(rule)}\n"
        
        return {
            'answer': answer,
            'question_type': 'when'
        }
    
    def _answer_where_question(self) -> Dict[str, Any]:
        
        return {
            'answer': 'El modelo analiza todas las características proporcionadas en los datos de entrada para tomar sus decisiones.',
            'question_type': 'where'
        }
    
    def _answer_what_question(self) -> Dict[str, Any]:
        
        return {
            'answer': f'El modelo es un árbol de decisión que clasifica datos en {len(self.class_names)} categorías: {", ".join(self.class_names)}.',
            'question_type': 'what'
        }
    
    def _answer_which_question(self) -> Dict[str, Any]:
        
        rules = self._extract_all_rules()
        if not rules:
            return {'answer': 'Ninguna regla disponible para comparación.', 'question_type': 'which'}
        
        best_rule = max(rules, key=lambda x: x.get('importance_score', 0.5))
        
        answer = f"La regla más importante es aquella que clasifica como '{best_rule['prediction']}' "
        answer += f"con {best_rule['confidence']:.0%} de confianza y {best_rule['samples']} muestras.\n\n"
        answer += f"Esta regla dice: {self._format_rule_text(best_rule)}"
        
        return {
            'answer': answer,
            'question_type': 'which',
            'best_rule': best_rule
        }
    
    def _answer_how_many_question(self) -> Dict[str, Any]:
        
        if self.training_data is None:
            return {'answer': 'Datos de entrenamiento no disponibles.', 'question_type': 'how_many'}
        
        total_samples = len(self.training_data)
        rules = self._extract_all_rules()
        
        answer = f"El modelo fue entrenado con {total_samples} muestras y posee {len(rules)} reglas de decisión.\n\n"
        
        if self.training_labels is not None:
            unique_labels, counts = np.unique(self.training_labels, return_counts=True)
            answer += "Distribución de las clases:\n"
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                answer += f"• {label}: {count} muestras ({percentage:.1f}%)\n"
        
        return {
            'answer': answer,
            'question_type': 'how_many',
            'total_samples': total_samples,
            'total_rules': len(rules)
        }
    
    def _answer_confidence_question(self, sample: np.ndarray) -> Dict[str, Any]:

        if sample is None:
            return {'answer': 'Proporcione una muestra para el análisis de confianza.', 'question_type': 'confidence'}
        
        explanation = self.generate_path_explanation(sample)
        confidence = explanation['confidence']
        
        if confidence >= 0.9:
            confidence_level = "muy alta"
            advice = "Esta es una decisión muy confiable."
        elif confidence >= 0.7:
            confidence_level = "alta"
            advice = "Esta es una decisión confiable."
        elif confidence >= 0.5:
            confidence_level = "moderada"
            advice = "Esta decisión tiene confianza moderada. Considere verificar los datos."
        else:
            confidence_level = "baja"
            advice = "Esta decisión tiene baja confianza. Se recomienda un análisis más detallado."
        
        answer = f"La confianza de esta clasificación es {confidence:.0%} ({confidence_level}).\n\n"
        answer += f"{advice}\n\n"
        answer += f"El modelo llegó a esta decisión mediante {len(explanation['path_steps'])} verificaciones secuenciales."
        
        return {
            'answer': answer,
            'question_type': 'confidence',
            'confidence_level': confidence_level,
            'confidence_value': confidence
        }
    
    def _answer_general_question(self, question: str) -> Dict[str, Any]:
        
        suggestions = [
            "¿Por qué se clasificó esta muestra así?",
            "¿Cómo funciona el modelo?",
            "¿Cuándo clasifica el modelo como positivo?",
            "¿Qué analiza el modelo?",
            "¿Cuál es la regla más importante?",
            "¿Cuántas muestras analizó el modelo?",
            "¿Cuál es la confianza de esta decisión?"
        ]
        
        answer = "Lo siento, no entendí su pregunta. Aquí hay algunas preguntas que puedo responder:\n\n"
        for i, suggestion in enumerate(suggestions, 1):
            answer += f"{i}. {suggestion}\n"
        
        answer += "\nIntente reformular su pregunta usando palabras clave como 'por qué', 'cómo', 'cuándo', 'qué', 'cuál' o 'cuántos'."
        
        return {
            'answer': answer,
            'question_type': 'general',
            'suggestions': suggestions
        }
