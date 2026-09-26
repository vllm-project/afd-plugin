# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Multi-pod AFD E2E runner.

Every pod runs the same program, derives its own slice of the topology from
shared inputs, and coordinates with its peers through a rendezvous store. No
process outside the pods holds test state.
"""
