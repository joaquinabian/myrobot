import unittest
from datetime import datetime
from unittest import mock

import robot_speech_client


class FakeSocket:
    def __init__(self, error=None, received=b""):
        self.error = error
        self.received = received
        self.messages = []
        self.closed = False

    def sendall(self, message):
        self.messages.append(message)
        if self.error:
            raise self.error

    def close(self):
        self.closed = True

    def recv(self, _size):
        data = self.received
        self.received = b""
        return data


class FakeInputStream:
    def __init__(self):
        self.calls = []

    def stop_stream(self):
        self.calls.append("stop")

    def start_stream(self):
        self.calls.append("start")


class SpeechClientTests(unittest.TestCase):
    def test_local_speech_is_enabled_by_default(self):
        with mock.patch("sys.argv", ["robot_speech_client.py"]):
            args = robot_speech_client.parse_args()

        self.assertTrue(args.echo)

    def test_no_echo_disables_local_speech(self):
        with mock.patch("sys.argv", ["robot_speech_client.py", "--no-echo"]):
            args = robot_speech_client.parse_args()

        self.assertFalse(args.echo)

    def test_piper_is_default_engine(self):
        with mock.patch("sys.argv", ["robot_speech_client.py"]):
            args = robot_speech_client.parse_args()

        self.assertEqual(args.tts_engine, "piper")
        self.assertEqual(
            args.tts_model,
            "voices/es_ES-davefx-medium.onnx",
        )

    def test_local_speech_stops_microphone_while_speaking(self):
        input_stream = FakeInputStream()
        source = mock.Mock()
        source.stream.pyaudio_stream = input_stream
        speaker = mock.Mock()

        robot_speech_client.say_local(
            speaker, "Has dicho hola", source
        )

        self.assertEqual(input_stream.calls, ["stop", "start"])
        speaker.say.assert_called_once_with("Has dicho hola")

    def test_send_does_not_repeat_after_socket_error(self):
        connection = robot_speech_client.ServerConnection("127.0.0.1", 65001)
        sock = FakeSocket(OSError("disconnected"))
        connection.sock = sock

        sent = connection.send("arriba")

        self.assertFalse(sent)
        self.assertEqual(sock.messages, [b"arriba\n"])
        self.assertTrue(sock.closed)
        self.assertIsNone(connection.sock)

    @mock.patch("robot_speech_client.select.select")
    def test_server_speech_is_received_as_complete_lines(self, select):
        connection = robot_speech_client.ServerConnection(
            "127.0.0.1", 65001
        )
        connection.sock = FakeSocket(
            received=b"say\tHola Joaquin\nsay\tMe aburro\n"
        )
        select.return_value = ([connection.sock], [], [])

        messages = connection.receive_speech()

        self.assertEqual(messages, ["Hola Joaquin", "Me aburro"])

    def test_server_speech_uses_local_speaker(self):
        connection = mock.Mock()
        connection.receive_speech.return_value = ["Hola Joaquin"]
        speaker = mock.Mock()
        source = mock.Mock()
        input_stream = FakeInputStream()
        source.stream.pyaudio_stream = input_stream

        robot_speech_client.play_server_speech(
            connection, speaker, source
        )

        speaker.say.assert_called_once_with("Hola Joaquin")
        self.assertEqual(input_stream.calls, ["stop", "start"])

    def test_command_gets_action_response(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "mira hacia la derecha", state, now=10.0
        )

        self.assertEqual(decision.intent, "command")
        self.assertEqual(decision.command, "derecha")
        self.assertEqual(decision.response, "Miro hacia la derecha")

    def test_follow_up_reuses_last_command(self):
        state = robot_speech_client.InteractionState()
        robot_speech_client.decide_interaction(
            "izquierda", state, now=10.0
        )

        decision = robot_speech_client.decide_interaction(
            "un poco más", state, now=11.0
        )

        self.assertEqual(decision.command, "izquierda")
        self.assertEqual(decision.response, "Miro hacia la izquierda")

    def test_follow_up_without_command_is_unknown(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "otra vez", state, now=10.0
        )

        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.command, "")
        self.assertEqual(decision.response, "Has dicho otra vez")

    def test_greeting_gets_short_response(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "Buenos días", state, now=10.0
        )

        self.assertEqual(decision.intent, "greeting")
        self.assertEqual(decision.response, "Hola, te escucho")

    def test_time_question_reads_raspberry_local_time(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "¿Qué hora es?",
            state,
            now=10.0,
            local_datetime=datetime(2026, 7, 30, 13, 5),
        )

        self.assertEqual(decision.intent, "time")
        self.assertEqual(decision.command, "")
        self.assertEqual(decision.response, "Son las 13 y 05")

    def test_one_oclock_uses_singular_response(self):
        response = robot_speech_client.local_time_response(
            datetime(2026, 7, 30, 1, 0)
        )

        self.assertEqual(response, "Es la 1 en punto")

    def test_identity_question_is_sent_to_server(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "¿Cómo me llamo?", state, now=10.0
        )

        self.assertEqual(decision.intent, "identity")
        self.assertEqual(decision.command, "quien_soy")
        self.assertEqual(decision.response, "")

    def test_recent_duplicate_does_not_speak_again(self):
        state = robot_speech_client.InteractionState()
        robot_speech_client.decide_interaction("hola", state, now=10.0)

        decision = robot_speech_client.decide_interaction(
            "Hola", state, now=12.0
        )

        self.assertEqual(decision.intent, "greeting")
        self.assertEqual(decision.response, "")

    def test_command_duplicate_keeps_command_but_suppresses_response(self):
        state = robot_speech_client.InteractionState()
        robot_speech_client.decide_interaction("derecha", state, now=10.0)

        decision = robot_speech_client.decide_interaction(
            "derecha", state, now=12.0
        )

        self.assertEqual(decision.command, "derecha")
        self.assertEqual(decision.response, "")

    def test_long_unknown_phrase_is_not_read_back(self):
        state = robot_speech_client.InteractionState()

        decision = robot_speech_client.decide_interaction(
            "esta es una frase bastante larga que no es una orden conocida",
            state,
            now=10.0,
        )

        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.response, "Te he oído")


if __name__ == "__main__":
    unittest.main()
