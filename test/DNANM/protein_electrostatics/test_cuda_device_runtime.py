#!/usr/bin/env python3

import math
import tempfile
from pathlib import Path

import _cuda_device_test_support as base


def require_device_signal(
    label,
    off,
    on,
):
    delta = base.max_state_delta(
        off["state"],
        on["state"],
    )

    print(
        "CUDA_DEVICE_SIGNAL",
        label,
        f"max_state_delta={delta:.16e}",
    )

    if not math.isfinite(delta):
        raise RuntimeError(
            f"{label}: non-finite device-state delta"
        )

    if off["state"] == on["state"]:
        raise RuntimeError(
            f"{label}: CUDA ON and OFF remain identical"
        )

    if delta <= 1.0e-14:
        raise RuntimeError(
            f"{label}: CUDA response below meaningful "
            f"configuration precision: {delta}"
        )


def main():

    expected_pd = base.dh_direct(
        -1.0
        / base.DNA_EFFECTIVE_CHARGE_E,
        1.02,
    )

    qred = (
        1.0
        / base.DNA_EFFECTIVE_CHARGE_E
    )

    expected_pp = base.dh_direct(
        qred * qred,
        0.8,
    )

    print(
        "EXPECTED_SYSTEM_PD",
        f"{expected_pd:.16e}",
    )

    print(
        "EXPECTED_SYSTEM_PP",
        f"{expected_pp:.16e}",
    )

    with tempfile.TemporaryDirectory(
        prefix="oxdna_cuda_device_"
    ) as td:

        root = Path(td)

        for mode in (
            "DNANM",
            "DNACT",
        ):

            # ====================================================
            # PD
            # ====================================================

            pd_off_a = base.run_case(
                root,
                f"{mode}_pd_off_a",
                mode=mode,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    0.0,
                ),
            )

            pd_off_b = base.run_case(
                root,
                f"{mode}_pd_off_b",
                mode=mode,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    0.0,
                ),
            )

            pd_on = base.run_case(
                root,
                f"{mode}_pd_on",
                mode=mode,
                pd=True,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    0.0,
                ),
            )

            base.require_deterministic_off(
                f"{mode}_PD",
                pd_off_a,
                pd_off_b,
            )

            base.require_host_energy(
                f"{mode}_PD",
                pd_off_a,
                pd_on,
                expected_pd,
            )

            require_device_signal(
                f"{mode}_PD",
                pd_off_a,
                pd_on,
            )

            # ====================================================
            # PP
            # ====================================================

            pp_off_a = base.run_case(
                root,
                f"{mode}_pp_off_a",
                mode=mode,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    +1.0,
                ),
            )

            pp_off_b = base.run_case(
                root,
                f"{mode}_pp_off_b",
                mode=mode,
                pd=False,
                pp=False,
                charges=(
                    +1.0,
                    0.0,
                    +1.0,
                ),
            )

            pp_on = base.run_case(
                root,
                f"{mode}_pp_on",
                mode=mode,
                pd=False,
                pp=True,
                charges=(
                    +1.0,
                    0.0,
                    +1.0,
                ),
            )

            base.require_deterministic_off(
                f"{mode}_PP",
                pp_off_a,
                pp_off_b,
            )

            base.require_host_energy(
                f"{mode}_PP",
                pp_off_a,
                pp_on,
                expected_pp,
            )

            require_device_signal(
                f"{mode}_PP",
                pp_off_a,
                pp_on,
            )

    print("CUDA_OFF_REPRODUCIBILITY=PASS")
    print("CUDA_HOST_PD_CONTROL=PASS")
    print("CUDA_HOST_PP_CONTROL=PASS")
    print("CUDA_DEVICE_PD_SIGNAL=PASS")
    print("CUDA_DEVICE_PP_SIGNAL=PASS")
    print("CUDA_DNANM_DNACT_DEVICE_SIGNAL=PASS")
    print("CUDA_PDPP_WIRING=PASS")


if __name__ == "__main__":
    main()
