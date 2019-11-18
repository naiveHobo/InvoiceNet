import sys
from invoicenet.acp.acp import AttendCopyParse

m = AttendCopyParse(sys.argv[1], restore='./models/acp/{}/best'.format(sys.argv[1]))
m.test_set()
