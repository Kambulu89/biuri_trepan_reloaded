"""Visualização de árvores: modelo visual (só leitura), layout hierárquico e renderer.

Camadas (nenhuma modifica a árvore científica):
    Tree Model --READ ONLY--> adapters/model -> layout -> render (Qt) -> viewport
Os módulos ``strings``, ``model``, ``layout`` e ``details`` não dependem de Qt.
"""
