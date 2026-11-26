# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

from afd_plugin.connectors.gpu.async_nvshmem.library import find_host_library


@pytest.mark.parametrize("explicit_home", [False, True])
def test_library_resolves_importable_wheel_or_configured_installation(
    tmp_path, monkeypatch, explicit_home
):
    user_site = tmp_path / "user-site"
    wheel = user_site / "nvidia" / "nvshmem" / "lib" / "libnvshmem_host.so.3"
    wheel.parent.mkdir(parents=True)
    wheel.touch()
    monkeypatch.syspath_prepend(user_site)
    monkeypatch.delenv("NVSHMEM_HOME", raising=False)
    expected = wheel
    if explicit_home:
        home = tmp_path / "nvshmem"
        expected = home / "lib" / "libnvshmem_host.so"
        expected.parent.mkdir(parents=True)
        expected.touch()
        monkeypatch.setenv("NVSHMEM_HOME", str(home))
    assert find_host_library() == expected
