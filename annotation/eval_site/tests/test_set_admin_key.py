import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import set_admin_key  # noqa: E402

SITE_DIR = Path(__file__).resolve().parents[1]


class SetAdminKeyTest(unittest.TestCase):
    def test_committed_files_never_carry_a_key(self):
        template = set_admin_key.TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(template.count(set_admin_key.PLACEHOLDER), 1)
        firebase = json.loads((SITE_DIR / "firebase.json").read_text(encoding="utf-8"))
        self.assertEqual(firebase["firestore"]["rules"], set_admin_key.DEPLOY_RULES.name)

    def test_render_fills_in_the_key_and_keeps_the_guard(self):
        rules = set_admin_key.render_rules(set_admin_key.TEMPLATE.read_text(encoding="utf-8"), "team-key-2026")
        self.assertIn("return 'team-key-2026';", rules)
        self.assertNotIn(set_admin_key.PLACEHOLDER, rules)
        self.assertIn("!adminKey().matches('__.*__')", rules)

    def test_unsafe_keys_are_rejected(self):
        for key in ["", "short", "has'quote", "has space", "__ADMIN_KEY__", "x" * 129]:
            with self.assertRaises(ValueError, msg=key):
                set_admin_key.render_rules("return '__ADMIN_KEY__';", key)

    def test_main_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "firestore.rules").write_text("function adminKey() { return '__ADMIN_KEY__'; }", encoding="utf-8")
            paths = {
                "SITE_DIR": tmp,
                "TEMPLATE": tmp / "firestore.rules",
                "DEPLOY_RULES": tmp / "firestore.deploy.rules",
                "LOCAL_KEY": tmp / ".local-admin.json",
            }
            with mock.patch.multiple(set_admin_key, **paths), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(set_admin_key.main(["team-key-2026"]), 0)
            self.assertIn("return 'team-key-2026';", (tmp / "firestore.deploy.rules").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((tmp / ".local-admin.json").read_text(encoding="utf-8")), {"key": "team-key-2026"})


if __name__ == "__main__":
    unittest.main()
