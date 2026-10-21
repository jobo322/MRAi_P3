"""Los scripts de `colab/scripts/` son copias de `scripts/`.

La convención está escrita en `scripts/cross_extract_fast.py` ("Any change in
StoP3 must be mirrored in BOTH cross_extract.py and this module"): el pipeline de
Colab usa las mismas fuentes, así que una corrección que se aplique solo en
`scripts/` deja Colab con el bug. Este test lo convierte en un fallo en vez de en
una divergencia silenciosa.

Si algún día Colab necesita divergir de verdad, quita el fichero de esta lista
(no lo dupliques con un cambio a medias).
"""
import filecmp
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRRORS = (
    "train_model.py",
    "preextract.py",
    "cross_extract.py",
    "cross_extract_fast.py",
    "jres_ingest.py",
    # módulos nuevos, también necesarios en Colab
    "hard_negatives.py",
    "weight_roles.py",
    "pair_marks.py",
)


class TestScriptMirrors(unittest.TestCase):
    def test_colab_scripts_match_scripts(self):
        for name in MIRRORS:
            with self.subTest(script=name):
                src = os.path.join(REPO_ROOT, "scripts", name)
                dst = os.path.join(REPO_ROOT, "colab", "scripts", name)
                self.assertTrue(os.path.isfile(src), f"falta scripts/{name}")
                self.assertTrue(os.path.isfile(dst), f"falta colab/scripts/{name}")
                self.assertTrue(
                    filecmp.cmp(src, dst, shallow=False),
                    f"colab/scripts/{name} está desincronizado con scripts/{name}: "
                    f"cópialo (cp scripts/{name} colab/scripts/)")


if __name__ == "__main__":
    unittest.main()
