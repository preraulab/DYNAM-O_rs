import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanitize_maturin_sbom as sanitizer


class SanitizeMaturinSbomTests(unittest.TestCase):
    def setUp(self):
        self.replacements = sanitizer.build_replacements(
            [("/Users/builder/work/DYNAM-O_toolbox", "/workspace")]
        )

    def test_preserves_bom_reference_identity(self):
        reference = (
            "path+file:///Users/builder/work/DYNAM-O_toolbox/"
            "DYNAM-O_rs/rust#dynamo_rs@0.1.0"
        )
        document = {
            "metadata": {"component": {"bom-ref": reference}},
            "dependencies": [{"ref": reference}],
        }
        rendered, changed = sanitizer.sanitize_json_bytes(
            json.dumps(document).encode(), self.replacements
        )
        sanitized = json.loads(rendered)

        self.assertTrue(changed)
        expected = "path+file:///workspace/DYNAM-O_rs/rust#dynamo_rs@0.1.0"
        self.assertEqual(sanitized["metadata"]["component"]["bom-ref"], expected)
        self.assertEqual(sanitized["dependencies"][0]["ref"], expected)

    def test_rejects_unmapped_home_path(self):
        document = {"bom-ref": "path+file:///home/another-user/project#crate@1.0.0"}
        with self.assertRaises(sanitizer.SanitizationError):
            sanitizer.sanitize_json_bytes(
                json.dumps(document).encode(), self.replacements
            )

    def test_wheel_rewrite_updates_record(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "example-0.1.0-py3-none-any.whl"
            sbom_name = "example-0.1.0.dist-info/sboms/example.cyclonedx.json"
            record_name = "example-0.1.0.dist-info/RECORD"
            module_name = "example.py"
            reference = (
                "path+file:///Users/builder/work/DYNAM-O_toolbox/"
                "DYNAM-O_rs/rust#dynamo_rs@0.1.0"
            )
            sbom = json.dumps({"bom-ref": reference}).encode()
            module = b"pass\n"
            record = (
                f"{module_name},,\n"
                f"{sbom_name},,\n"
                f"{record_name},,\n"
            ).encode()
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(module_name, module)
                archive.writestr(sbom_name, sbom)
                archive.writestr(record_name, record)

            count = sanitizer.sanitize_wheel(
                wheel, self.replacements, check_only=False
            )
            self.assertEqual(count, 1)
            sanitizer.sanitize_wheel(wheel, self.replacements, check_only=True)

            with zipfile.ZipFile(wheel) as archive:
                sanitized = json.loads(archive.read(sbom_name))
                self.assertNotIn("/Users/", sanitized["bom-ref"])
                rows = {
                    row[0]: row
                    for row in csv.reader(
                        io.StringIO(archive.read(record_name).decode())
                    )
                }
                content = archive.read(sbom_name)
                self.assertEqual(
                    rows[sbom_name][1], sanitizer._record_digest(content)
                )
                self.assertEqual(rows[sbom_name][2], str(len(content)))
                self.assertEqual(rows[record_name][1:], ["", ""])

    def test_installed_sbom_rewrite_updates_record(self):
        with tempfile.TemporaryDirectory() as directory:
            site_packages = Path(directory)
            dist_info = site_packages / "example-0.1.0.dist-info"
            sbom = dist_info / "sboms" / "example.cyclonedx.json"
            record = dist_info / "RECORD"
            sbom.parent.mkdir(parents=True)
            reference = (
                "path+file:///Users/builder/work/DYNAM-O_toolbox/"
                "DYNAM-O_rs/rust#dynamo_rs@0.1.0"
            )
            sbom.write_text(json.dumps({"bom-ref": reference}))
            sbom_name = sbom.relative_to(site_packages).as_posix()
            record_name = record.relative_to(site_packages).as_posix()
            record.write_text(f"{sbom_name},,\n{record_name},,\n")

            count = sanitizer.sanitize_json_target(
                sbom.parent, self.replacements, check_only=False
            )
            self.assertEqual(count, 1)
            sanitizer.sanitize_json_target(
                sbom.parent, self.replacements, check_only=True
            )

            rows = {
                row[0]: row
                for row in csv.reader(io.StringIO(record.read_text()))
            }
            content = sbom.read_bytes()
            self.assertEqual(rows[sbom_name][1], sanitizer._record_digest(content))
            self.assertEqual(rows[sbom_name][2], str(len(content)))
            self.assertEqual(rows[record_name][1:], ["", ""])


if __name__ == "__main__":
    unittest.main()
