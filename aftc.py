import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m=2.0; g=9.81
Ixx=Iyy=0.02166667; Izz=0.04
L=0.174; kf=8.54858e-6; km=0.016
wmax=1500.0; tu=0.0125; td=0.025
Ir=2.65e-5; kr=0.01
rho=1.225; Cd=1.3; Ac=0.005
dt=0.002
S=np.array([1,1,-1,-1.0])
PX=np.array([L,-L,L,-L]); PY=np.array([L,-L,-L,L])
M=np.vstack([np.ones(4),-PY,PX,S*km])
Tmax=kf*wmax**2
lT=0.5; Ncnt=10
Kp=3.0; Kd=3.2; Katt=8.0; Kw=15.0; Kn=4.0

def Rm(f,t,p):
    cf,sf,ct,st,cp,sp=np.cos(f),np.sin(f),np.cos(t),np.sin(t),np.cos(p),np.sin(p)
    return np.array([[ct*cp,sf*st*cp-cf*sp,cf*st*cp+sf*sp],
                     [ct*sp,sf*st*sp+cf*cp,cf*st*sp-sf*cp],
                     [-st,sf*ct,cf*ct]])

def f(x,wc):
    v=x[3:6]; ph,th,ps=x[6:9]; p,q,r=x[9:12]; om=x[12:16]
    R=Rm(ph,th,ps)
    Ti=kf*om**2
    drag=-0.5*Cd*Ac*rho*v*np.abs(v)
    a=np.array([0,0,g])-R[:,2]*Ti.sum()/m+drag/m
    tt=M[1:]@Ti
    wr=S@om
    omd=np.where(wc>om,(wc-om)/tu,(wc-om)/td)
    wrd=S@omd
    wd=np.array([((Iyy-Izz)*q*r+tt[0]+Ir*q*wr)/Ixx,
                 ((Izz-Ixx)*r*p+tt[1]-Ir*p*wr)/Iyy,
                 ((Ixx-Iyy)*p*q+tt[2]+Ir*wrd-kr*r)/Izz])
    sph,cph,tth,cth=np.sin(ph),np.cos(ph),np.tan(th),np.cos(th)
    eul=np.array([p+q*sph*tth+r*cph*tth,q*cph-r*sph,(q*sph+r*cph)/cth])
    return np.concatenate([v,a,eul,wd,omd])

class SOF:
    def __init__(s,n,wn=20.0,z=0.8):
        s.wn=wn; s.z=z; s.x=None; s.d=np.zeros(n)
    def step(s,u):
        if s.x is None: s.x=u.copy()
        a=s.wn**2*(u-s.x)-2*s.z*s.wn*s.d
        s.d=s.d+a*dt
        s.x=s.x+s.d*dt
        return s.x,s.d

class Ctrl:
    def __init__(s,nref=(0.0,0.0)):
        s.Fg=SOF(3); s.Fw=SOF(1); s.Fo=SOF(4); s.Fwr=SOF(1)
        s.oh=np.full(4,np.sqrt(m*g/(4*kf))); s.wc=s.oh.copy()
        s.alive=np.ones(4,bool); s.cnt=np.zeros(4,int)
        s.i0=None; s.tdet=None; s.lh=np.zeros(4); s.nref=np.array(nref)
    def step(s,t,gy,vz,att,pos,vel,pr,vr,ar):
        ph,th,ps=att
        s.oh+=dt*(s.wc-s.oh)/np.where(s.wc>s.oh,tu,td)
        if s.i0 is not None: s.oh[s.i0]=0.0
        gf,gd=s.Fg.step(gy)
        _,wd=s.Fw.step(np.array([vz]))
        o2f,_=s.Fo.step(s.oh**2)
        wrf,wrd=s.Fwr.step(np.array([S@s.oh]))
        p,q,r=gf
        if s.i0 is None:
            tau_hat=np.array([Ixx*gd[0]-((Iyy-Izz)*q*r+Ir*q*wrf[0]),
                              Iyy*gd[1]-((Izz-Ixx)*r*p-Ir*p*wrf[0]),
                              Izz*gd[2]-((Ixx-Iyy)*p*q+Ir*wrd[0]-kr*r)])
            T_hat=m*(g-wd[0])/(np.cos(ph)*np.cos(th))
            A=M*(kf*o2f)[None,:]
            s.lh=1-np.linalg.solve(A,np.concatenate([[T_hat],tau_hat]))
            s.cnt=np.where(s.lh>=lT,s.cnt+1,0)
            if s.cnt.max()>=Ncnt:
                s.i0=int(np.argmax(s.lh)); s.alive[s.i0]=False
                s.oh[s.i0]=0.0; s.tdet=t
        ad=Kp*(pr-pos)+Kd*(vr-vel)+ar
        fv=np.array([0,0,g])-ad
        R=Rm(ph,th,ps)
        Tc=m*fv[2]/max(np.cos(ph)*np.cos(th),0.5)
        u=fv/np.linalg.norm(fv)
        tau0=M[1:]@(kf*o2f)
        if s.i0 is None:
            cp,sp=np.cos(ps),np.sin(ps)
            uxp=u[0]*cp+u[1]*sp; uyp=-u[0]*sp+u[1]*cp
            phd=np.arcsin(np.clip(-uyp,-1,1)); thd=np.arctan2(uxp,u[2])
            psd=(0-ps+np.pi)%(2*np.pi)-np.pi
            phdot=Katt*(phd-ph); thdot=Katt*(thd-th); psdot=Katt*psd
            wd_=np.array([phdot-psdot*np.sin(th),
                          thdot*np.cos(ph)+psdot*np.sin(ph)*np.cos(th),
                          -thdot*np.sin(ph)+psdot*np.cos(ph)*np.cos(th)])
            tc=tau0+np.array([Ixx,Iyy,Izz])*(Kw*(wd_-gy)-gd)
            Ti=np.linalg.solve(M,np.concatenate([[Tc],tc]))
        else:
            n=R.T@u
            nu=-Kn*(n[:2]-s.nref)
            nz=max(n[2],0.3)
            c=np.array([n[1],-n[0]])*gy[2]
            pqd=np.array([[0,1/nz],[-1/nz,0]])@(nu-c)
            nup=Kw*(pqd-gy[:2])
            txy=tau0[:2]+np.array([Ixx,Iyy])*(nup-gd[:2])
            Ma=M[:3][:,s.alive]
            Ta=np.linalg.solve(Ma,np.concatenate([[Tc],txy]))
            Ti=np.zeros(4); Ti[s.alive]=Ta
        s.wc=np.sqrt(np.clip(Ti,0,Tmax)/kf)
        s.wc[~s.alive]=0.0
        return s.wc

def ref(t):
    ax,wx=0.8,0.4; ay,wy=0.5,0.8; az,wz=0.3,0.6
    pr=np.array([ax*(1-np.cos(wx*t)),ay*(1-np.cos(wy*t)),-2-az*(1-np.cos(wz*t))])
    vr=np.array([ax*wx*np.sin(wx*t),ay*wy*np.sin(wy*t),-az*wz*np.sin(wz*t)])
    ar=np.array([ax*wx**2*np.cos(wx*t),ay*wy**2*np.cos(wy*t),-az*wz**2*np.cos(wz*t)])
    return pr,vr,ar

def run(i_f=0,tf=10.0,T=40.0,seed=0,nref=(0.0,0.0)):
    rng=np.random.default_rng(seed)
    x=np.zeros(16); x[2]=-2.0; x[12:16]=np.sqrt(m*g/(4*kf))
    c=Ctrl(nref); ok=np.ones(4,bool)
    N=int(T/dt); log=np.zeros((N,1+3+3+4+4+3))
    for k in range(N):
        t=k*dt
        if t>=tf and ok[i_f]: ok[i_f]=False; x[12+i_f]=0.0
        pr,vr,ar=ref(t)
        gy=x[9:12]+rng.normal(0,8.7e-4,3)
        vz=x[5]+rng.normal(0,0.01)
        wc=c.step(t,gy,vz,x[6:9],x[0:3],x[3:6],pr,vr,ar)
        wc=np.where(ok,wc,0.0)
        k1=f(x,wc);k2=f(x+0.5*dt*k1,wc);k3=f(x+0.5*dt*k2,wc);k4=f(x+dt*k3,wc)
        x=x+dt/6*(k1+2*k2+2*k3+k4)
        if not ok[i_f]: x[12+i_f]=0.0
        log[k]=np.concatenate([[t],x[0:3],pr,x[12:16],c.lh,x[9:12]])
    return log,c

if __name__=="__main__":
    log,c=run(i_f=0)
    t=log[:,0]
    print("detected rotor:",c.i0,"at t=",c.tdet)
    post=t>(c.tdet+8 if c.tdet else 1e9)
    e=log[:,1:4]-log[:,4:7]
    print("rms pos err post-recovery:",np.sqrt((e[post]**2).mean(0)) if post.any() else None)
    print("max rotor speed:",log[:,7:11].max(),"final yaw rate:",log[-1,-1])

    def save(name,draw):
        plt.figure(figsize=(8,4))
        draw()
        plt.xlabel("t (s)"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(name,dpi=130); plt.close()

    for j,nm in enumerate("xyz"):
        def d(j=j,nm=nm):
            plt.plot(t,log[:,1+j],label="response")
            plt.plot(t,log[:,4+j],"--",label="reference")
            plt.axvline(10.0,c="r",ls=":",label="fault")
            if c.tdet: plt.axvline(c.tdet,c="g",ls=":",label="detected")
            plt.ylabel(nm+" (m)"); plt.legend()
        save("pos_"+nm+".png",d)

    def d():
        for i in range(4): plt.plot(t,log[:,11+i],label="l%d"%(i+1))
        plt.axhline(lT,c="k",ls="--",label="threshold")
        plt.ylim(-1.5,1.5); plt.ylabel("LOE estimate"); plt.legend()
    save("loe.png",d)

    def d():
        for i in range(4): plt.plot(t,log[:,7+i],label="rotor %d"%(i+1))
        plt.axhline(wmax,c="k",ls="--",label="limit")
        plt.ylabel("rotor speed (rad/s)"); plt.legend()
    save("rotors.png",d)

    def d():
        plt.plot(t,log[:,-1]); plt.ylabel("yaw rate r (rad/s)")
    save("yaw_rate.png",d)

    plt.figure(figsize=(5,5))
    plt.plot(log[:,1],log[:,2],label="response")
    plt.plot(log[:,4],log[:,5],"--",label="reference")
    plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.axis("equal"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("xy_path.png",dpi=130); plt.close()
