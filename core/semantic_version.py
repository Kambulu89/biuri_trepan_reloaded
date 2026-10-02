"""Versão do pipeline semântico.

Entra nas chaves de cache: qualquer alteração à geração/seleção de features
ontológicas deve incrementar este valor para que nenhum modelo treinado com a
versão anterior seja reutilizado.
"""
SEMANTIC_PIPELINE_VERSION = "9.3.0"
