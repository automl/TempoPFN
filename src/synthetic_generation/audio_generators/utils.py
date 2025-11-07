import os
import tempfile
import time
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from pyo import NewTable, Server, TableRec


def run_offline_pyo(
    synth_builder: Callable[[], object],
    server_duration: float,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    """
    Render a pyo synthesis graph offline and return a numpy waveform.

    Parameters
    ----------
    synth_builder : Callable[[], object]
        Function that builds and returns a pyo object representing the synth graph.
    server_duration : float
        Duration in seconds to run the offline server.
    sample_rate : int
        Sample rate for the offline server.
    length : int
        Number of samples to return.

    Returns
    -------
    np.ndarray
        Waveform of shape (length,).
    """
    # Suppress pyo console messages during offline rendering
    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        s = Server(sr=sample_rate, nchnls=1, duplex=0, audio="offline")
        # Use a unique temp filename to avoid clashes across concurrent jobs
        tmp_wav = os.path.join(
            tempfile.gettempdir(),
            f"pyo_offline_{os.getpid()}_{int(time.time_ns())}.wav",
        )
        # The filename is required by pyo's offline server even if we record to a table
        s.recordOptions(dur=server_duration, filename=tmp_wav, fileformat=0)
        s.boot()

        table = NewTable(length=server_duration, chnls=1)

        synth_obj = synth_builder()

        # Record the output of the synth object to the table
        _ = TableRec(synth_obj, table, fadetime=0.01).play()

        s.start()
        # Offline mode runs immediately to completion; no need for sleep
        s.stop()
        s.shutdown()
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            # Best-effort cleanup; ignore errors
            pass

    waveform = np.array(table.getTable())
    if waveform.size > length:
        waveform = waveform[:length]
    elif waveform.size < length:
        # Pad with zeros if the rendered buffer is shorter than requested
        pad = np.zeros(length - waveform.size, dtype=waveform.dtype)
        waveform = np.concatenate([waveform, pad], axis=0)

    return waveform


def normalize_waveform(values: np.ndarray) -> np.ndarray:
    """
    Normalize a waveform to have max absolute value of 1 (if nonzero).
    """
    max_abs = np.max(np.abs(values)) if values.size > 0 else 0.0
    if max_abs > 0:
        return values / max_abs
    return values
