def test_test():
    import demo
    import demo_graph
    import demo_embedding

    import subprocess
    subprocess.check_output(['flake8','tensorboardX'])
    