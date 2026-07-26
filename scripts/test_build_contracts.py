from pathlib import Path
import re
import unittest


REPO = Path(__file__).resolve().parent.parent


class BuildContractTests(unittest.TestCase):
    def test_library_build_excludes_static_archive(self):
        manifest = (REPO / "rust" / "Cargo.toml").read_text()
        lib_section = re.search(r"(?ms)^\[lib\]\n(.*?)(?=^\[)", manifest)

        self.assertIsNotNone(lib_section)
        self.assertIn('crate-type = ["cdylib", "rlib"]', lib_section.group(1))
        self.assertNotIn('"staticlib"', lib_section.group(1))


if __name__ == "__main__":
    unittest.main()
