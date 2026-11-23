# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async CAM startup, sessions, Attention handoff/layout and FFN execution."""

# TokenNums_Rankid_Layeridx from compact dispatch-recv carries the layer in word 2.
AFD_ASYNC_LAYER_INDEX = 2
