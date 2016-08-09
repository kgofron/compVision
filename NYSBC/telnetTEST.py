import telnetlib

HOST = "192.168.1.35"

tn = telnetlib.Telnet(HOST)
"""tn.write("MG \"hello\"\r")
tn.write("MT*=?\r") 
tn.write("RS\r")

print tn.read_until(":")[:-2] #read_some()
print tn.read_until(":")[:-2] #read_some()
"""

tn.write("SH\r")
print "SH " +tn.read_some()
tn.write("JGA=2000\r")
print "JGA " + tn.read_some()
tn.write("BGA\r")
print "BGA " + tn.read_some()
tn.write("WT 1000\r")
print "WT " + tn.read_some()
tn.write("ST\r")
print "ST " + tn.read_some()
tn.write("RPA\r")
VAL = tn.read_some()
tn.write("MO\r")
print "MO " + tn.read_some()


print "VAL" + VAL[:-2]
tn.close()
