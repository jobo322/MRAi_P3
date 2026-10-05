import os, gc, argparse
import numpy as np
import nmrglue as ng

def main(P_dirc,S_dirc,PN="PN", thres=0.1,fromat='Pipe',out=None):
    if fromat == 'Pipe':
        dic_p, P = ng.pipe.read(P_dirc)
        dic_s, S = ng.pipe.read(S_dirc)
    else:
        dic_p, P = ng.fileio.bruker.read_pdata(P_dirc)
        dic_s, S = ng.fileio.bruker.read_pdata(S_dirc)
    if dic_p == dic_s:
        dic = dic_p
    else:
        raise ValueError("P3 and ft3 must be for same spectrum.")
    if fromat != 'Pipe': P *= 1/np.max(np.abs(P))
    P3 = np.abs(P) if PN == "PN" else np.array(P)

    msep = [1,1] if P3.ndim == 2 else [1,1,1]
    loc = np.array(ng.peakpick.find_all_thres_fast(P3, thres=thres, msep=msep))
    P5 = np.hstack([loc, np.zeros((loc.shape[0], 2))])
    
    idx = tuple(loc.T)
    P5[:, -1] = S[idx] / np.median(np.abs(S))
    P5[:, -2] = 100 * P[idx]
    
    acc = np.zeros((loc.shape[1],loc.shape[0]))
    for i in range(loc.shape[1]):
        loc_P = np.array(loc); loc_P[:,i] += 1; loc_P[loc_P[:, i] > P.shape[i]-1,i] = 0; idx_P = tuple(loc_P.T)
        loc_N = np.array(loc); loc_N[:,i] -= 1; loc_N[loc_N[:, i] < 0,i] = P.shape[i]-1; idx_N = tuple(loc_N.T)
        acc[i] = 0.5*(np.abs(P[idx_P])-np.abs(P[idx_N]))/(2*np.abs(P[idx])-np.abs(P[idx_P])-np.abs(P[idx_N]))
    if fromat == 'Pipe':
        P5[:,-3] = dic['FDF2ORIG']/dic['FDF2OBS'] + (dic['FDSIZE'] - (idx[-1]+acc[-1]) - 1) * (dic['FDF2SW']/dic['FDF2OBS'])/dic['FDSIZE']
        P5[:,-4] = dic['FDF1ORIG']/dic['FDF1OBS'] + (dic['FDF1FTSIZE'] - (idx[-2]+acc[-2]) - 1) * (dic['FDF1SW']/dic['FDF1OBS'])/dic['FDF1FTSIZE']
        if P3.ndim == 3: P5[:,-5] = dic['FDF3ORIG']/dic['FDF3OBS'] + (dic['FDF3FTSIZE'] - (idx[-3]+acc[-3]) - 1) * (dic['FDF3SW']/dic['FDF3OBS'])/dic['FDF3FTSIZE']
        if out is None: out = os.path.splitext(S_dirc)[0] + '.P5'
        np.save(os.path.splitext(out)[0] + '_P5.npy',P5)
        np.savetxt(out, P5, delimiter=' ', fmt='%8.4f')
    else:
        P5[:,-3] = dic['procs']['OFFSET'] - ((idx[-1]+acc[-1])/dic['procs']['SI'] ) * ( dic['procs']['SW_p']/dic['procs']['SF'] )
        P5[:,-4] = dic['proc2s']['OFFSET'] - ((idx[-2]+acc[-2])/dic['proc2s']['SI'] ) * ( dic['proc2s']['SW_p']/dic['proc2s']['SF'] )
        if P3.ndim == 3: P5[:,-5] = dic['proc3s']['OFFSET'] - ( (idx[-3]+acc[-3])/dic['proc3s']['SI'] ) * ( dic['proc3s']['SW_p']/dic['proc3s']['SF'] )
        if out is None: out = os.path.join(os.path.join(P_dirc), 'P5.npy')
        np.save(out,P5)
    del dic, dic_p, dic_s, P3, P, S, P5
    gc.collect()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate P5 peak list from spectrum and its P3.")
    ap.add_argument("-p3", help="Path to P3 file (Pipe or Bruker pdata).")
    ap.add_argument("-spec", required=True, help="Path to spectrum file (Pipe or Bruker pdata).")
    ap.add_argument("-pn", choices=["PN", "P"], default="P", help="PN - use Pos and Neg  peaks")
    ap.add_argument("-pt", type=float, default=0.1, help="Threshold for probability thereshould (default: 0.1).")
    ap.add_argument("-ft", choices=["Pipe", "Bruker"], default="Pipe", help="Format of the spectrum (Bruker or Pipe).")
    ap.add_argument("-o", default=None, help="Output file path (optional).")
    args = ap.parse_args()
    main(args.p3, args.spec, PN=args.pn, thres=args.pt, fromat=args.ft, out=args.o)