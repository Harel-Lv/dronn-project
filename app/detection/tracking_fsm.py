class TrackingCommandStabilizer:
    """Small FSM-style stabilizer for tracking commands."""

    def __init__(self, stable_frames_required: int = 3, no_target_frames_required: int = 3):
        self.stable_frames_required = max(1, stable_frames_required)
        self.no_target_frames_required = max(1, no_target_frames_required)
        self.displayed_command = "NO_TARGET"
        self._candidate_command: str | None = None
        self._candidate_count = 0
        self._no_target_count = 0

    def update(self, command: str) -> str:
        if command == "NO_TARGET":
            self._no_target_count += 1
            if self._no_target_count >= self.no_target_frames_required:
                self.displayed_command = "NO_TARGET"
                self._candidate_command = None
                self._candidate_count = 0
            return self.displayed_command
        self._no_target_count = 0

        if command == self.displayed_command:
            self._candidate_command = None
            self._candidate_count = 0
            return self.displayed_command

        if command == self._candidate_command:
            self._candidate_count += 1
            if self._candidate_count >= self.stable_frames_required:
                self.displayed_command = command
                self._candidate_command = None
                self._candidate_count = 0
            return self.displayed_command

        self._candidate_command = command
        self._candidate_count = 1
        return self.displayed_command
