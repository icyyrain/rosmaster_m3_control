import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pc_client.red_rl import run_session


class RunSessionTests(unittest.TestCase):
    def test_failed_dry_run_never_starts_real_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(run_session.subprocess, 'run', return_value=mock.Mock(returncode=1)) as run:
                with mock.patch('builtins.input', return_value='q'):
                    self.assertFalse(run_session.collect_one(Path(directory), None))
            self.assertEqual(run.call_count, 1)
            self.assertNotIn('--execute', run.call_args.args[0])

    def test_failed_real_trial_stops_instead_of_continuing(self):
        with tempfile.TemporaryDirectory() as directory:
            statuses = [mock.Mock(returncode=0), mock.Mock(returncode=1)]
            with mock.patch.object(run_session.subprocess, 'run', side_effect=statuses) as run:
                with self.assertRaisesRegex(RuntimeError, 'Inspect the arm'):
                    run_session.collect_one(Path(directory), None)
            self.assertEqual(run.call_count, 2)
            self.assertIn('--execute', run.call_args.args[0])

    def test_success_reports_new_episode_without_record_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)

            def invoke(command, **_kwargs):
                if '--execute' in command:
                    episode = dataset / 'new'
                    episode.mkdir()
                    (episode / 'episode.json').write_text(json.dumps({
                        'human_success': True, 'steps': 24,
                    }), encoding='utf-8')
                return mock.Mock(returncode=0)

            with mock.patch.object(run_session.subprocess, 'run', side_effect=invoke) as run:
                self.assertTrue(run_session.collect_one(dataset, None))
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all('--record' not in call.args[0] for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
