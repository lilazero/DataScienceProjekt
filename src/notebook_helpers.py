"""
notebook_helpers.py

Hilfsfunktion, um die nummerierten Skripte (z.B. '01_load_data.py') als
Python-Module zu importieren. Normale Python-Module dürfen nicht mit einer
Ziffer beginnen, daher reicht ein einfaches `import 01_load_data` nicht aus.
Diese Funktion lädt die Datei trotzdem direkt über ihren Pfad.

Wird ausschließlich im Jupyter-Notebook (run_all.ipynb) verwendet, damit die
Dateinamen der Skripte unverändert (mit führender Nummer) bleiben können.
"""

import importlib.util
import sys
import os


def load_step(filename: str):
    """
    Lädt ein Skript wie '01_load_data.py' als Python-Modul und führt es aus.
    Gibt das Modul zurück, sodass z.B. step.main() aufgerufen werden kann.
    """
    module_name = os.path.splitext(filename)[0]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
