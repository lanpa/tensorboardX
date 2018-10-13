from tensorboardX import SummaryWriter
from datetime import datetime

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter()

for i in range(100):
    writer.add_text('TAGHERE', 'testStringHere' + str(i), i) # use global step

writer.close() # close the writer

