# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project


class Graph:
    """Captured device graph double counting replays and resets."""

    def __init__(self):
        self.replay_count = 0
        self.reset_count = 0

    def replay(self):
        self.replay_count += 1

    def reset(self):
        self.reset_count += 1
