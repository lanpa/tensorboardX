# called by demo_global_writer

from tensorboardX import GlobalSummaryWriter

writer = GlobalSummaryWriter.getSummaryWriter()

writer.add_text('my_log', 'greeting from global1')

for i in range(100):
    writer.add_scalar('global1', i)

for i in range(100):
    writer.add_scalar('common', i)