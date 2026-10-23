# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

from afd_plugin.ffn_faas.settings import FaaSOptions


@pytest.mark.parametrize("value", [0, -1, True, "invalid"])
def test_slot_settings_require_positive_integers(value: int | str) -> None:
    with pytest.raises((ValueError, TypeError)):
        FaaSOptions.from_mapping({"slots": value})


def test_legacy_cam_quantization_cannot_silently_select_faas() -> None:
    with pytest.raises(ValueError, match="unknown"):
        FaaSOptions.from_mapping({"dynamicQuant": 1})
