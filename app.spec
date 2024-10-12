# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')

# datas += [('./utils.py',".")]
# datas += [('./slicer2.py',".")]



a = Analysis(
    ['app.py',
    ],
    pathex=['/Users/liuyue/Downloads/FunAsr_Faster_Whisper_Multi_Subs'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    module_collection_mode={ 'gradio': 'py'}
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisper_turbo',
    # icon='AnyConv.com__paints_logo.icns',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

a.datas += Tree('./faster-whisper-large-v3-turbo-ct2', prefix='faster-whisper-large-v3-turbo-ct2')
a.datas += Tree('./models_from_modelscope', prefix='models_from_modelscope')
a.datas += Tree('./output', prefix='output')
# a.datas += Tree('./bin', prefix='bin')


# a.datas += Tree('./output', prefix='output')



coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whisper_turbo',
)


