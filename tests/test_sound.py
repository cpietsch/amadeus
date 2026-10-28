import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from amadeus import sound


class AmadeusTest(unittest.TestCase):
    def test_notes(self):
        self.assertAlmostEqual(sound._note('A4'), 440.0)
        self.assertAlmostEqual(sound._note('A5'), 880.0)
        self.assertAlmostEqual(sound._note('C#4'), sound._note('Db4'))

    def test_wav_files(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'XDG_CACHE_HOME': d}):
            for kind in ('jingle', 'error'):
                p = sound.wav(kind)
                self.assertEqual(p, Path(d) / 'amadeus' / f'{kind}.wav')
                with wave.open(str(p)) as w:
                    secs = w.getnframes() / w.getframerate()
                self.assertTrue(0.5 < secs < 3, secs)

    def test_failed(self):
        self.assertTrue(sound.failed({'error': 'refused'}))
        self.assertTrue(sound.failed({'events': ['FAILED 1234: FAILED (exit 1:0, 3 s)']}))
        self.assertFalse(sound.failed({'events': ['CANCELLED 1234: CANCELLED', 'ENDED 5: COMPLETED']}))
        self.assertFalse(sound.failed({'error': ''}))
        self.assertFalse(sound.failed('text'))

    def test_off_and_rate_limit(self):
        with mock.patch.object(sound.threading, 'Thread') as t:
            with mock.patch.dict(os.environ, {'AMADEUS_SOUND': 'off'}):
                sound.play('jingle')
            t.assert_not_called()
            with mock.patch.dict(os.environ, {'AMADEUS_SOUND': 'on'}):
                sound._last_error = 0.0
                sound.play('error')
                sound.play('error')          # within ERROR_GAP_S: dropped
            self.assertEqual(t.call_count, 1)

    def test_no_player_is_silent(self):
        with mock.patch.object(sound.shutil, 'which', return_value=None), \
             mock.patch.object(sound, '_wsl', return_value=False):
            self.assertIsNone(sound._player(Path('/x.wav')))


if __name__ == '__main__':
    unittest.main()
