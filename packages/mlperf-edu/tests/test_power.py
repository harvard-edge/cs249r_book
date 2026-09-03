import time

from mlperf.power import PowerMeter


def test_power_meter_reports_estimated_energy():
    meter = PowerMeter(nominal_watts=10.0)
    meter.start()
    time.sleep(0.001)
    report = meter.stop_report()

    assert report["source"] == "estimated_nominal"
    assert report["average_watts"] == 10.0
    assert report["duration_seconds"] > 0
    assert report["energy_joules"] > 0
