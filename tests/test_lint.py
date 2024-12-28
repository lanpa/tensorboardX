def test_linting():
    import subprocess
    subprocess.check_output(['ruff', 'check', 'tensorboardX'])
