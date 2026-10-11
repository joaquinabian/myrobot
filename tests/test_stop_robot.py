import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
STOP_SCRIPT = PROJECT_DIR / "stop_robot.sh"


class StopRobotTests(unittest.TestCase):
    def run_script(self, running_pids=""):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bin_dir = temp_path / "bin"
            proc_dir = temp_path / "proc"
            outside_dir = temp_path / "other-project"
            bin_dir.mkdir()
            proc_dir.mkdir()
            outside_dir.mkdir()
            log_path = temp_path / "kill.log"
            pgrep = bin_dir / "pgrep"
            kill = bin_dir / "mock-kill"
            pgrep.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *robot_server.py*)\n"
                "    mkdir -p \"$PROC_ROOT/$STOP_SCRIPT_PID\"\n"
                "    ln -s \"$TEST_PROJECT_DIR\" "
                "\"$PROC_ROOT/$STOP_SCRIPT_PID/cwd\"\n"
                "    printf 'python\\0robot_server.py\\0' > "
                "\"$PROC_ROOT/$STOP_SCRIPT_PID/cmdline\"\n"
                "    printf '%s\\n' 101 102 \"$STOP_SCRIPT_PID\"\n"
                "    ;;\n"
                "  *robot_video_client.py*) printf '%s\\n' 103 ;;\n"
                "  *robot_speech_client.py*) printf '%s\\n' 104 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            kill.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$KILL_LOG\"\n"
                "if [ \"$1\" = -0 ]; then\n"
                "  case \" $RUNNING_PIDS \" in *\" $2 \"*) exit 0 ;; esac\n"
                "  exit 1\n"
                "fi\n",
                encoding="utf-8",
            )
            pgrep.chmod(0o755)
            kill.chmod(0o755)

            processes = {
                101: (PROJECT_DIR, ["python", "robot_server.py"]),
                102: (outside_dir, ["python", "robot_server.py"]),
                103: (outside_dir,
                      ["python", str(PROJECT_DIR / "robot_video_client.py")]),
                104: (PROJECT_DIR, ["python", "./robot_speech_client.py"]),
            }
            for pid, (cwd, arguments) in processes.items():
                pid_dir = proc_dir / str(pid)
                pid_dir.mkdir()
                (pid_dir / "cwd").symlink_to(cwd, target_is_directory=True)
                (pid_dir / "cmdline").write_bytes(
                    b"\0".join(arg.encode() for arg in arguments) + b"\0"
                )

            env = os.environ.copy()
            env.update({
                "PATH": f"{bin_dir}:{env['PATH']}",
                "KILL_COMMAND": str(kill),
                "KILL_LOG": str(log_path),
                "PROC_ROOT": str(proc_dir),
                "RUNNING_PIDS": running_pids,
                "STOP_TIMEOUT": "0",
                "TEST_PROJECT_DIR": str(PROJECT_DIR),
            })
            result = subprocess.run(
                ["bash", str(STOP_SCRIPT)],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
            calls = (log_path.read_text(encoding="utf-8").splitlines()
                     if log_path.exists() else [])
            return result, calls

    def test_only_sends_sigterm_to_project_processes(self):
        result, calls = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        term_calls = [call for call in calls if call.startswith("-TERM")]
        self.assertEqual(term_calls, ["-TERM 101", "-TERM 103", "-TERM 104"])
        self.assertNotIn("-TERM 102", term_calls)

    def test_excludes_stop_script_pid(self):
        result, calls = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        term_pids = {call.split()[1] for call in calls
                     if call.startswith("-TERM")}
        self.assertEqual(term_pids, {"101", "103", "104"})

    def test_reports_processes_that_ignore_sigterm_without_sigkill(self):
        result, calls = self.run_script(running_pids="103")

        self.assertEqual(result.returncode, 1)
        self.assertIn("still running after SIGTERM", result.stderr)
        self.assertIn("103", result.stderr)
        self.assertTrue(all("-KILL" not in call and "-9" not in call
                            for call in calls))


if __name__ == "__main__":
    unittest.main()
