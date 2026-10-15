import ast
from pathlib import Path
import unittest
from repo.kaggle.jobs.avbind_native_pairs import recording_key


class IndependentNativePairTests(unittest.TestCase):
    def test_other_clips_of_same_original_recording_are_excluded_together(self):
        first='temporal/106/xhXcJ6bhX2w/a.mp4';second='perception/19/xhXcJ6bhX2w/b.mp4'
        self.assertEqual(recording_key(first),recording_key(second))
        self.assertEqual(recording_key(first),recording_key(first.replace('/','\\')))

    def test_pair_feature_and_annotation_have_no_query_label_clock_access(self):
        tree=ast.parse(Path('repo/kaggle/jobs/avbind_native_pairs.py').read_text(encoding='utf-8'))
        for fn in (n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('native_pair_')):
            names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
            self.assertFalse(names & {'question','answer_options','gold','task','start','interval','video_path'})
            # Text feature extraction has a local character-span `end`, not a
            # supplied media clock; only media/mode or content are API inputs.
            self.assertTrue({arg.arg for arg in fn.args.args} <= {'media','mode','content'})

    def test_selection_never_uses_benchmark_answers_or_questions(self):
        source=Path('server_jobs/avbind_native_pair_select.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        keys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)}
        self.assertFalse(keys & {'question','answer','task_type_abbr','data_type'})
