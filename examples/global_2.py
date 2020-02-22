# called by demo_global_writer

from tensorboardX import GlobalSummaryWriter

writer = GlobalSummaryWriter.getSummaryWriter()

writer.add_text('my_log', 'greeting from global2')

for i in range(100):
    writer.add_scalar('global2', i)

for i in range(100):
    writer.add_scalar('common', i)