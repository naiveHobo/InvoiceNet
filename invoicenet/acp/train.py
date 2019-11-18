import sys
import trainer
from invoicenet.acp.acp import AttendCopyParse

print("Training...")
model = trainer.train(AttendCopyParse(sys.argv[1]))
