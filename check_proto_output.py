from tensorboardX import summary
import google.protobuf.text_format as text_format

hp = {'lr': 0.1}
mt = {'accuracy': 0.1}
exp, ssi, sei = summary.hparams(hp, mt)

print("--- exp ---")
print(text_format.MessageToString(exp))
print("--- ssi ---")
print(text_format.MessageToString(ssi))
print("--- sei ---")
print(text_format.MessageToString(sei))

print("--- tuple str ---")
print(str((exp, ssi, sei)))
