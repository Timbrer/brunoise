from multiprocessing import Event, Queue
from lightparam import Param
from lightparam.param_qt import ParametrizedQt
from scanning import (
    Scanner,
    ScanningParameters,
    ScanningState,
    ImageReconstructor,
)
from pathlib import Path
from streaming_save import StackSaver, SavingParameters, SavingStatus
from arrayqueues.shared_arrays import ArrayQueue
from queue import Empty
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Optional
from time import sleep
import numpy as np

PIEZO_MAX_UM = 450.0
PIEZO_UM_PER_VOLT = 45.0
PIEZO_MAX_VOLTAGE = PIEZO_MAX_UM / PIEZO_UM_PER_VOLT

class ExperimentSettings(ParametrizedQt):
    def __init__(self):
        super().__init__()
        self.name = "recording"
        self.n_planes = Param(1, (1, 500))
        self.n_frames = Param(100, (1, 100000))
        self.dz = Param(1.0, (-50, 50.0), unit="um")
        self.save_dir = Param(str(Path.home() / "Desktop"), gui=False)


class ScanningSettings(ParametrizedQt):
    def __init__(self):
        super().__init__()
        self.name = "scanning"
        self.aspect_ratio = Param(1.0, (0.2, 5.0))
        self.voltage = Param(3.0, (0.2, 5.0))
        self.framerate = Param(2.0, (0.1, 10.0))
        self.binning = Param(10, (1, 50))
        self.output_rate_khz = Param(400, (50, 2000))
        self.n_turn = Param(10, (0, 100))
        self.n_extra_init = Param(100, (0, 100))
        self.pause = Param(1, (0, 1))  # Int as Boolean GUI generation is not supported.


def convert_params(st: ScanningSettings, piezo_z_um=0.0) -> ScanningParameters:
    """
    Converts the GUI scanning settings in parameters appropriate for the
    laser scanning

    """
    pause = True if st.pause else False

    sample_rate = st.output_rate_khz * 1000
    n_total = sample_rate / st.framerate
    # Loosens the restraint by 2 * the turn value, as the first and last line require one turn less.
    n_total += 2 * st.n_turn

    # Solving for the biggest image surface is basically a constraint problem of the form:
    # ax**2 + bx + c = 0, where a is the aspect ratio (can be seen as y * x where y = a * x), b is the two turns,
    # and c is the total number of available positions (given the desired frequency and sampling rate).
    b = 2 * st.n_turn
    if pause:  # If pause is enabled, additional points dependent on x will be added to the trajectory.
        b += 1
    n = (-b + np.sqrt(b**2 - (4 * st.aspect_ratio * -n_total))) / (2 * st.aspect_ratio) # Image dimensions.

    # Change the y-axis to get the right aspect ratio.
    n_x = int(np.floor(n))
    n_y = int(np.floor(n * st.aspect_ratio))

    # No need to get rid of 2 turns, we already added it before.
    n_extra = int(n_total - ((n_x * b) + (n_x*n_y)))

    mystery_offset = -int(round(st.output_rate_khz * 0.8))
    voltage_max = st.voltage
    if st.aspect_ratio >= 1:
        voltage_y = voltage_max
        voltage_x = voltage_y / st.aspect_ratio
    else:
        voltage_y = voltage_max * st.aspect_ratio
        voltage_x = voltage_max
    voltage_z = float(np.clip(piezo_z_um / PIEZO_UM_PER_VOLT, 0.0, PIEZO_MAX_VOLTAGE))

    sp = ScanningParameters(
        voltage_x=voltage_x,
        voltage_y=voltage_y,
        voltage_z=voltage_z,
        n_x=n_x,
        n_y=n_y,
        n_turn=st.n_turn,
        n_extra=n_extra,
        sample_rate_out=sample_rate,
        mystery_offset=mystery_offset,
        framerate=st.framerate,
        pause=pause
    )
    return sp


class ExperimentState(QObject):
    sig_scanning_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.experiment_start_event = Event()
        self.scanning_settings = ScanningSettings()
        self.experiment_settings = ExperimentSettings()
        self.pause_after = False

        self.end_event = Event()
        self.scanner = Scanner(self.experiment_start_event)
        self.scanning_parameters = None
        self.reconstructor = ImageReconstructor(
            self.scanner.data_queue, self.scanner.stop_event
        )
        self.save_queue = ArrayQueue(max_mbytes=800)
        self.timestamp_queue = Queue()

        self.saver = StackSaver(
            self.scanner.stop_event, self.save_queue, self.timestamp_queue
        )
        self.save_status: Optional[SavingStatus] = None

        self.piezo_z_um = 225.0
        self.recording_start_z_um = 0.0
        self.scanning_settings.sig_param_changed.connect(self.send_scan_params)
        self.scanning_settings.sig_param_changed.connect(self.send_save_params)
        self.scanner.start()
        self.reconstructor.start()
        self.saver.start()
        self.open_setup()

        self.paused = False
        self.recording = False
        self.current_plane = 0
        self.frames_in_plane = 0
        self.plane_end_requested = False
        self.recording_n_frames = None
        self.recording_n_planes = None
        self.recording_plane_z_um = None

    @property
    def saving(self):
        return self.recording

    @property
    def save_in_progress(self):
        return self.saver.busy_signal.is_set()

    def open_setup(self):
        self.send_scan_params()

    def start_experiment(self, first_plane=True):
        if first_plane and self.save_in_progress:
            return False

        if first_plane:
            self.current_plane = 0
            self.recording_n_frames = self.experiment_settings.n_frames
            self.recording_n_planes = self.experiment_settings.n_planes
            self.recording_start_z_um = self.piezo_z_um
            self.recording_plane_z_um = self.compute_plane_z_positions(
                self.recording_start_z_um,
                self.recording_n_planes,
                self.experiment_settings.dz,
            )

        params_to_send = convert_params(self.scanning_settings, self.piezo_z_um)
        params_to_send.scanning_state = ScanningState.EXPERIMENT_RUNNING
        params_to_send.n_frames = self.recording_n_frames
        self.scanner.parameter_queue.put(params_to_send)

        self.recording = True
        self.frames_in_plane = 0
        self.plane_end_requested = False
        if first_plane:
            self.send_save_params()
            self.saver.saving_signal.set()
        self.experiment_start_event.set()
        return True

    def end_experiment(self, force=False):
        self.plane_end_requested = True
        self.experiment_start_event.clear()

        if not force and self.current_plane + 1 < self.recording_n_planes:
            self.advance_plane()
        else:
            self.recording = False
            sleep(0.2)
            self.saver.saving_signal.clear()
            if self.pause_after:
                self.pause_scanning()
            else:
                self.restart_scanning()

    def restart_scanning(self):
        params_to_send = convert_params(self.scanning_settings, self.piezo_z_um)
        params_to_send.scanning_state = ScanningState.PREVIEW
        self.scanner.parameter_queue.put(params_to_send)
        self.paused = False

    def pause_scanning(self):
        params_to_send = convert_params(self.scanning_settings, self.piezo_z_um)
        params_to_send.scanning_state = ScanningState.PAUSED
        self.scanner.parameter_queue.put(params_to_send)
        self.paused = True

    def advance_plane(self):
        self.current_plane += 1
        self.piezo_z_um = self.recording_plane_z_um[self.current_plane]
        sleep(0.2)
        self.start_experiment(first_plane=False)

    def close_setup(self):
        """ Cleanup on programe close:
        end all parallel processes, close all communication channels

        """
        self.scanner.stop_event.set()
        self.end_event.set()
        self.scanner.join()
        self.reconstructor.join()
        self.saver.join()

    def get_image(self):
        try:
            images = -self.reconstructor.output_queue.get(timeout=0.001)
            try:
                t = self.scanner.time_queue.get(timeout=0.001)
            except Empty:
                t = 0
                print("scanner time queue is empty")
            if self.recording:
                self.save_queue.put(images)
                self.timestamp_queue.put(t)
                self.frames_in_plane += 1
                if (
                    not self.plane_end_requested
                    and self.frames_in_plane >= self.recording_n_frames
                ):
                    self.end_experiment()
            return images
        except Empty:
            return None

    def set_piezo_z_um(self, z_um):
        self.piezo_z_um = float(np.clip(z_um, 0.0, PIEZO_MAX_UM))
        if not self.saving:
            self.send_scan_params()

    def compute_plane_z_positions(self, start_z_um, n_planes, dz_um):
        return tuple(
            float(np.clip(start_z_um + (i * dz_um), 0.0, PIEZO_MAX_UM))
            for i in range(n_planes)
        )

    def send_scan_params(self):
        self.scanning_parameters = convert_params(self.scanning_settings, self.piezo_z_um)
        self.scanner.parameter_queue.put(self.scanning_parameters)
        self.reconstructor.parameter_queue.put(self.scanning_parameters)
        self.sig_scanning_changed.emit()

    def send_save_params(self):
        if self.recording or self.experiment_start_event.is_set():
            n_t = self.recording_n_frames
            n_z = self.recording_n_planes
            plane_z_um = self.recording_plane_z_um
        else:
            n_t = self.experiment_settings.n_frames
            n_z = self.experiment_settings.n_planes
            plane_z_um = self.compute_plane_z_positions(
                self.piezo_z_um,
                n_z,
                self.experiment_settings.dz,
            )

        self.saver.saving_parameter_queue.put(
            SavingParameters(
                output_dir=Path(self.experiment_settings.save_dir),
                plane_size=(self.scanning_parameters.n_x, self.scanning_parameters.n_y),
                n_t=n_t,
                n_z=n_z,
                plane_z_um=plane_z_um,
            )
        )

    def get_save_status(self) -> Optional[SavingStatus]:
        try:
            self.save_status = self.saver.saved_status_queue.get(timeout=0.001)
            return self.save_status
        except Empty:
            pass
        return None
