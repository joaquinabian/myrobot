import unittest
from types import SimpleNamespace

from robot_server import (
    FaceMotionController,
    LatestVideoState,
    ServoController,
    VideoControlThread,
    limited_delta,
)


class FakeServo:
    def __init__(self):
        self.angle = None


class FakeServoController:
    def __init__(self):
        self.angle_1 = 121
        self.angle_2 = 73
        self.moves = []

    def move_by(self, delta_1=0, delta_2=0, smooth=False):
        self.angle_1 += delta_1
        self.angle_2 += delta_2
        self.moves.append((delta_1, delta_2))


class FaceMotionControllerTests(unittest.TestCase):
    def make_controller(self, alpha=0.5):
        return FaceMotionController(
            filter_alpha=alpha,
            deadband_x=35,
            deadband_y=35,
            pixels_per_degree_x=45,
            pixels_per_degree_y=45,
            max_step_x=2,
            max_step_y=3,
        )

    def test_deadband_produces_no_step(self):
        controller = self.make_controller()

        filtered_x, filtered_y, step_x, step_y = controller.update(35, -20)

        self.assertEqual((filtered_x, filtered_y), (35.0, -20.0))
        self.assertEqual((step_x, step_y), (0, 0))

    def test_filter_uses_previous_error(self):
        controller = self.make_controller(alpha=0.25)
        controller.update(100, -100)

        filtered_x, filtered_y, _, _ = controller.update(0, 0)

        self.assertEqual(filtered_x, 75.0)
        self.assertEqual(filtered_y, -75.0)

    def test_step_is_proportional_and_limited_per_axis(self):
        controller = self.make_controller(alpha=1.0)

        _, _, step_x, step_y = controller.update(500, -500)

        self.assertEqual(step_x, 2)
        self.assertEqual(step_y, -3)

    def test_reset_discards_previous_error(self):
        controller = self.make_controller(alpha=0.25)
        controller.update(100, 100)
        controller.reset()

        filtered_x, filtered_y, _, _ = controller.update(0, 0)

        self.assertEqual((filtered_x, filtered_y), (0.0, 0.0))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_controller(alpha=0)


class LimitedDeltaTests(unittest.TestCase):
    def test_delta_is_limited_in_both_directions(self):
        self.assertEqual(limited_delta(90, 120, 2), 2)
        self.assertEqual(limited_delta(90, 60, 2), -2)

    def test_delta_reaches_target_without_overshoot(self):
        self.assertEqual(limited_delta(89, 90, 2), 1)
        self.assertEqual(limited_delta(90, 90, 2), 0)


class ServoControllerTests(unittest.TestCase):
    def test_start_angles_and_axis_limits_are_independent(self):
        servo_1 = FakeServo()
        servo_2 = FakeServo()
        controller = ServoController(
            servo_1,
            servo_2,
            min_angle_1=20,
            max_angle_1=140,
            min_angle_2=10,
            max_angle_2=160,
            start_angle_1=121,
            start_angle_2=73,
        )

        self.assertEqual((servo_1.angle, servo_2.angle), (121, 73))

        controller.set_angles(1, 179, smooth=False)

        self.assertEqual((servo_1.angle, servo_2.angle), (20, 160))
        self.assertEqual((controller.angle_1, controller.angle_2), (20, 160))


class TargetRecoveryTests(unittest.TestCase):
    def make_thread(self):
        thread = VideoControlThread.__new__(VideoControlThread)
        thread.args = SimpleNamespace(
            video_recovery_timeout=2.0,
            video_print_interval=10.0,
        )
        thread.servos = FakeServoController()
        thread.motion = SimpleNamespace(reset=lambda: None)
        thread.last_delta_1 = 1
        thread.last_delta_2 = -2
        thread.last_print_time = 100.0
        thread.recovery_finished = False
        return thread

    def test_last_movement_continues_during_recovery_window(self):
        thread = self.make_thread()

        recovered = thread.maybe_recover_target(now=11.9, msg_time=10.0)

        self.assertTrue(recovered)
        self.assertEqual(thread.servos.moves, [(1, -2)])

    def test_recovery_stops_after_timeout(self):
        thread = self.make_thread()

        recovered = thread.maybe_recover_target(now=12.1, msg_time=10.0)

        self.assertFalse(recovered)
        self.assertEqual(thread.servos.moves, [])
        self.assertTrue(thread.recovery_finished)

    def test_no_recovery_when_last_command_was_stationary(self):
        thread = self.make_thread()
        thread.last_delta_1 = 0
        thread.last_delta_2 = 0

        recovered = thread.maybe_recover_target(now=11.9, msg_time=10.0)

        self.assertFalse(recovered)
        self.assertEqual(thread.servos.moves, [])


class LatestVideoStateTests(unittest.TestCase):
    def test_connection_state_is_included_in_snapshot(self):
        state = LatestVideoState()
        self.assertFalse(state.snapshot()[-1])

        state.client_connected()
        self.assertTrue(state.snapshot()[-1])

        state.client_disconnected()
        self.assertFalse(state.snapshot()[-1])


if __name__ == "__main__":
    unittest.main()
