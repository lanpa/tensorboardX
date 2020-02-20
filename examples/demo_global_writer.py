# This program show that you can use summary writer globally
# So that you can use the writer like the python.logging module

# This file triggers global_1 and global_2 to do their job.
import global_1
import time
time.sleep(2)
import global_2

