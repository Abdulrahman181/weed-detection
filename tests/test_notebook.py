import ast
import json
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "weed-detection.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def split_function_from_notebook():
    notebook = load_notebook()
    source = "".join(notebook["cells"][6]["source"])
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "split_originals_with_augmentations"
    )

    def deterministic_split(images, test_size, random_state):
        if test_size != 0.2 or random_state != 42:
            raise AssertionError("Unexpected split configuration")
        return images[:-1], images[-1:]

    namespace = {"re": re, "train_test_split": deterministic_split}
    isolated_module = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated_module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["split_originals_with_augmentations"]


class NotebookMaintenanceTests(unittest.TestCase):
    def test_code_cells_have_python_syntax(self):
        for index, cell in enumerate(load_notebook()["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            python_lines = [
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("!", "%"))
            ]
            if "\n" in source:
                python_source = "\n".join(python_lines)
                if python_source.strip():
                    with self.subTest(cell=index):
                        ast.parse(python_source)

    def test_setup_does_not_delete_kaggle_working_directory(self):
        setup = "".join(load_notebook()["cells"][0]["source"])
        self.assertNotIn("shutil.rmtree", setup)
        self.assertNotIn("os.unlink", setup)
        self.assertIn("Path(\"/kaggle/working\")", setup)

    def test_notebook_execution_outputs_are_cleared(self):
        for cell in load_notebook()["cells"]:
            if cell.get("cell_type") == "code":
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_augmentations_stay_with_their_original_split(self):
        split_images = split_function_from_notebook()
        originals = ["field_a.jpg", "field_b.jpg", "field_c.jpg", "field_d.jpg"]
        augmented = ["aug_0_field_a.jpg", "aug_1_field_b.jpg", "aug_0_field_d.jpg"]

        train_images, validation_images = split_images(originals, augmented)

        self.assertEqual(
            set(train_images),
            {"field_a.jpg", "field_b.jpg", "field_c.jpg", "aug_0_field_a.jpg", "aug_1_field_b.jpg"},
        )
        self.assertEqual(
            set(validation_images), {"field_d.jpg", "aug_0_field_d.jpg"}
        )
        self.assertTrue(set(train_images).isdisjoint(validation_images))

    def test_unmatched_augmentation_fails_closed(self):
        split_images = split_function_from_notebook()
        with self.assertRaisesRegex(ValueError, "Cannot map augmented image"):
            split_images(["field_a.jpg", "field_b.jpg"], ["aug_0_unknown.jpg"])


if __name__ == "__main__":
    unittest.main()
