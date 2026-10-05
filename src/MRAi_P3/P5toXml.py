import os, argparse, sys
import numpy as np
import nmrglue as ng
from datetime import date

def parse_bruker_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 5 or parts[-2] != "pdata":
        raise ValueError("Not a valid Bruker pdata directory")
    current_datetime = date.today()
    current_datetime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    return {"name":  parts[-4],"expno": parts[-3],"procno": parts[-1],"date": current_datetime}

def main(P_dirc,out=None):
    try:
        P5 = np.load(os.path.join(P_dirc, 'P5.npy'))
    except Exception as e:
        sys.exit(f"Error reading P5 file: {e}")
    info = parse_bruker_path(P_dirc); Dim = P5.shape[1] - 2
    XML = f'''<?xml version="1.0" encoding="UTF-8"?>
<PeakList modified="{info['date']}">
  <PeakList{Dim}D>
    <PeakList{Dim}DHeader creator="MRAi_P5" date="{info['date']}" expNo="{info['expno']}" name="{info['name']}_P5" owner="MRAi" procNo="{info['procno']}" source="">
      <PeakPickDetails>
</PeakPickDetails>
    </PeakList{Dim}DHeader>
'''
    if Dim == 2:
        body = '    <Peak2D F1="%s" F2="%s" intensity="%s" type="%s"/>\n'
        for element in P5:
            XML += body% (element[0],element[1],element[2],0)
    if Dim == 3:
        body = '    <Peak3D F1="%s" F2="%s" F3="%s" intensity="%s" type="%s"/>\n'
        for element in P5:
            XML += body% (element[0],element[1],element[2],element[3],0)
    XML += f'  </PeakList{Dim}D>\n</PeakList>'
    if out is not None:
        outFile = open(os.path.join(out, "peaklist.xml"),'w'); outFile.write(XML); outFile.close()
    out = os.path.join(os.path.join(P_dirc), "peaklist.xml")
    outFile = open(out,'w'); outFile.write(XML); outFile.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert P5 numeric table to TopSpin XML peaklist.")
    ap.add_argument("p5", help="Path to input .P5 file")
    ap.add_argument("-o", default=None, help="Output file path (optional).")
    args = ap.parse_args()
    main(args.p5,out=args.o)