"""Per-corner entry diagnosis from ROAR_LOG run.csv files (Task 8c).

  python tools/corner_diag.py logs/<label>_*.run.csv

Per corner (apex line_s; braking zones of the lap-2 profile), medians over laps (lap 1 launch skipped):
vref_err  v - v_ref at turn-in; maxout  max outward offset (m); start_rel_ti  where the monotone drift to it began,
relative to turn-in (m); brake_frac  share of the drift with a_cmd < -3; beta  body sideslip yaw - course (rad, +
into the turn) at max drift; ff  the sideslip feed-forward k*a_lat there; kcmd_kref, kyaw_kcmd  medians over the drift;
aerr  median |mean accel of the next 4 ticks - a_cmd| while braking; jump  max |da_cmd| per tick; vexit  v - v_ref at
apex + 80 m. Body yaw: the logged yaw column, else e_psi + feed-forward (legacy logs).
"""
import sys, collections, numpy as np
import os
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,R+'/competition_code')
from track import Track
from vehicle import Vehicle
tr=Track(np.load(R+'/competition_code/data/racing_line.npz')['xy']); L=tr.length
veh=Vehicle.from_json(R+'/competition_code/data/vehicle_params.json')
DT=0.05
CORNERS=[(795,937),(1281,1329),(1619,1697),(2545,2695),(3719,3835),(3935,3993),(4093,4113),(5239,5413)]
psiu=np.unwrap(tr.psi)
I=lambda x,a: np.interp(x,tr.s,a,period=L)
rows=collections.defaultdict(list)
for f in sys.argv[1:]:
    d=np.genfromtxt(f,delimiter=',',names=True)
    ls,v=d['line_s'],d['v']; al=np.array([veh.a_lat(x) for x in ls])
    ff=veh.p['sideslip_k']*np.clip(v**2*d['kappa_ref'],-al,al)
    psi=I(ls,psiu)
    yaw=np.unwrap(d['yaw']) if 'yaw' in d.dtype.names else psi+d['e_psi']+ff
    xy=np.c_[I(ls,tr.xy[:,0]),I(ls,tr.xy[:,1])]+d['e_y'][:,None]*np.c_[-np.sin(psi),np.cos(psi)]
    dxy=np.gradient(xy,axis=0); chi=np.unwrap(np.arctan2(dxy[:,1],dxy[:,0]))
    beta=(yaw-chi+np.pi)%(2*np.pi)-np.pi
    kyaw=np.gradient(np.unwrap(yaw))/DT/np.maximum(v,1)
    ar=np.r_[np.convolve(np.diff(v)/DT,np.ones(4)/4,'valid'),np.zeros(4)]  # mean accel over ticks k..k+3
    for sb,sa in CORNERS:
        sg=np.sign(I(sa,tr.kappa))
        dist=(ls-(sb-20))%L
        idx=np.flatnonzero(dist<(sa-sb)+80)
        for lap in np.split(idx,np.flatnonzero(np.diff(idx)>5)+1):
            if len(lap)<20 or (d['s_total'][lap[0]]<300): continue
            out=-sg*d['e_y'][lap]; kr=sg*d['kappa_ref'][lap]
            km=int(np.argmax(out))
            st=km
            while st>0 and out[st-1]<out[st] and out[st]>0.3: st-=1   # drift start: where the monotone rise to the max began
            ti=int(np.argmax(kr>0.5*kr.max()))
            brk=d['a_cmd'][lap]< -3
            w=slice(st,max(km,st+1))
            phase=('brake' if brk[st] else 'nobrake')+('+turn' if kr[st]>0.3*kr.max() else '')
            kc=sg*d['kappa_cmd'][lap]
            rows[sa].append(dict(
                vref_err=float((v-d['v_ref'])[lap][ti]),
                maxout=float(out[km]),
                start_rel_ti=float(ls[lap][st]-ls[lap][ti]),
                phase=phase,
                brake_frac=float(brk[w].mean()),
                beta=float(sg*beta[lap][km]), ff=float(sg*ff[lap][km]),
                kcmd_kref=float(np.median(kc[w])/max(np.median(kr[w]),1e-4)),
                kyaw_kcmd=float(np.median(sg*kyaw[lap][w])/max(np.median(kc[w]),1e-4)),
                aerr=float(np.median(np.abs(ar[lap]-d['a_cmd'][lap])[brk])) if brk.any() else np.nan,
                jump=float(np.abs(np.diff(d['a_cmd'][lap])).max()),
                stab=float(d['a_cmd'][lap][km:].min()) if km<len(lap)-1 else np.nan,
                vexit=float((v-d['v_ref'])[lap][-1])))
keys=['vref_err','maxout','start_rel_ti','brake_frac','beta','ff','kcmd_kref','kyaw_kcmd','aerr','jump','vexit']
print('| apex line_s | n | '+' | '.join(keys)+' | drift-start phase |')
print('|'+'---|'*(len(keys)+3))
for sa,rs in sorted(rows.items()):
    ph=collections.Counter(r['phase'] for r in rs).most_common(1)[0]
    print('| %d | %d | '%(sa,len(rs))+' | '.join('%.2f'%np.nanmedian([r[k] for r in rs]) for k in keys)+' | %s (%d/%d) |'%(ph[0],ph[1],len(rs)))
