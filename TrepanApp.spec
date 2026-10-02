# -*- mode: python ; coding: utf-8 -*-
import os
from dtreeviz import __path__ as dtreeviz_path
from PyInstaller.utils.hooks import collect_all

owlready_datas, owlready_binaries, owlready_hiddenimports = collect_all('owlready2')

block_cipher = None

# --- Encontrar o caminho para os executáveis do Graphviz ---
# Normalmente fica em "C:\\Program Files\\Graphviz\\bin"
# Este código tenta encontrar, mas se falhar, você pode definir o caminho manualmente
graphviz_bin_path = ''
for path in os.environ['PATH'].split(os.pathsep):
    if 'graphviz' in path.lower() and 'bin' in path.lower():
        graphviz_bin_path = path
        break

if not graphviz_bin_path:
    # Se não encontrar no PATH, defina manualmente aqui:
    # graphviz_bin_path = 'C:\\Program Files\\Graphviz\\bin'
    raise Exception("Caminho do Graphviz não encontrado. Adicione-o ao PATH ou defina manualmente no arquivo .spec")


# --- Análise principal da aplicação ---
a = Analysis(['run_biuri.py'],
             pathex=[os.path.abspath('.')],
             
             # Incluir os executáveis do Graphviz
             binaries=[(os.path.join(graphviz_bin_path, '*.exe'), 'graphviz'),
                       (os.path.join(graphviz_bin_path, '*.dll'), 'graphviz')]
                      + owlready_binaries,
             
             # Incluir arquivos de dados necessários
             datas=[
                # Adicionar as fontes do dtreeviz
                (os.path.join(dtreeviz_path[0], 'fonts'), 'dtreeviz/fonts')
             ] + owlready_datas,
             
             # Incluir módulos que o PyInstaller pode não ver
             hiddenimports=['PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtGui', 'PyQt6.QtCore']
                           + owlready_hiddenimports,
             
             # Caminhos para hooks adicionais (não necessários para PyQt6)
             hookspath=[],
             
             # Hooks de tempo de execução (não necessários para PyQt6)
             runtime_hooks=[],
             
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [], # Os binários e dados já são gerenciados pelo Analysis
          [], #
          exclude_binaries=True,
          name='TrepanApp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,  # Sem console
          icon=None ) # Você pode adicionar um ícone aqui: icon='caminho/para/seu/icone.ico'

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TrepanApp')
