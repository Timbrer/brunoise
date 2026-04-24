from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brunoise"))

from scanning import (  # noqa: E402
    NI_USB_6363_MAX_AI_SAMPLE_RATE,
    ScanningParameters,
    sample_rate_in,
    signal_delay_samples,
)
from state import ScanningSettings, convert_params  # noqa: E402


def test_default_scanning_settings_match_usb_6363_safe_defaults():
    settings = ScanningSettings()

    params = convert_params(settings)

    assert params.sample_rate_out == 100000.0
    assert params.n_bin == 5
    assert signal_delay_samples(params) == -40
    assert sample_rate_in(params) == 500000.0
    assert params.pause is True


def test_default_scanning_parameters_match_gui_defaults():
    params = ScanningParameters()

    assert params.sample_rate_out == 100000.0
    assert params.n_bin == 5
    assert params.n_extra == 100
    assert signal_delay_samples(params) == -40
    assert params.pause is True


def test_output_rate_is_limited_by_input_rate_and_binning():
    settings = ScanningSettings()
    settings.output_rate_khz = 1000.0
    settings.binning = 5

    params = convert_params(settings)

    assert params.sample_rate_out == NI_USB_6363_MAX_AI_SAMPLE_RATE / 5
    assert sample_rate_in(params) == NI_USB_6363_MAX_AI_SAMPLE_RATE
